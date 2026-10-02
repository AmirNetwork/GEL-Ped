# Author: Amir Ghorbani
"""Compact interaction-transformer baseline for pedestrian velocity prediction.

This module intentionally reuses the graph baseline's causal, goal-aligned
wall-free node and neighbour arrays so the two backbones differ in architecture
rather than information.  Neighbours are unordered tokens processed by masked
self-attention with a learned summary token.  A separate focal-node MLP is
fused with that summary to predict a correction to the observed-velocity
anchor.
"""

from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from pedgeom.benchmarks import local_target, local_target_from_vectors
from pedgeom.calibration import VelocitySamples
from pedgeom.graph_baseline import _graph_arrays

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:  # pragma: no cover - optional dependency guidance
    raise ImportError(
        "The interaction-transformer baseline requires PyTorch; install the "
        "CPU build listed in requirements-graph.txt."
    ) from error


def _transformer_arrays(
    samples: VelocitySamples,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the graph protocol's exact causal wall-free inputs."""

    return _graph_arrays(samples, include_wall=False)


@dataclass(frozen=True)
class _PreparedArrays:
    node: np.ndarray
    edge: np.ndarray
    mask: np.ndarray
    target: np.ndarray
    weights: np.ndarray


def _prepare_batches(
    batches: Sequence[VelocitySamples], *, residual: bool
) -> _PreparedArrays:
    """Concatenate complete runs and give every run equal aggregate mass."""

    if not batches:
        raise ValueError("at least one complete-run batch is required")
    node_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for batch in batches:
        if len(batch.target) == 0:
            raise ValueError("complete-run batches must not be empty")
        node, edge, mask = _transformer_arrays(batch)
        anchor = local_target_from_vectors(batch, batch.features[:, 0])
        node_parts.append(node)
        edge_parts.append(edge)
        mask_parts.append(mask)
        target_parts.append(local_target(batch) - anchor if residual else local_target(batch))
        weight_parts.append(np.full(len(batch.target), 1.0 / len(batch.target)))
    node = np.concatenate(node_parts)
    edge = np.concatenate(edge_parts)
    mask = np.concatenate(mask_parts)
    target = np.concatenate(target_parts)
    weights = np.concatenate(weight_parts)
    # A constant rescaling leaves the objective unchanged but keeps average
    # batch weights close to one, matching the graph baseline numerically.
    weights *= len(weights) / np.sum(weights)
    return _PreparedArrays(node, edge, mask, target, weights)


class _InteractionTransformer(nn.Module):
    """Permutation-invariant masked Transformer over neighbour tokens."""

    def __init__(
        self,
        *,
        node_features: int,
        edge_features: int = 7,
        width: int = 32,
        num_heads: int = 4,
        num_layers: int = 1,
        feedforward_multiplier: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if width < 2 or width % num_heads:
            raise ValueError("width must be at least two and divisible by num_heads")
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        if feedforward_multiplier < 1:
            raise ValueError("feedforward_multiplier must be at least one")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.edge_projection = nn.Sequential(
            nn.Linear(edge_features, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.summary_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.normal_(self.summary_token, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=num_heads,
            dim_feedforward=feedforward_multiplier * width,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.neighbour_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )
        self.output_head = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.Linear(width, 2),
        )

    def forward(
        self,
        node: torch.Tensor,
        edge: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        edge_tokens = self.edge_projection(edge)
        summary = self.summary_token.expand(edge.shape[0], -1, -1)
        tokens = torch.cat((summary, edge_tokens), dim=1)
        # The summary token is always valid, which also makes an empty
        # neighbourhood well-defined rather than an all-masked attention row.
        padding_mask = torch.cat(
            (
                torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device),
                ~mask,
            ),
            dim=1,
        )
        encoded = self.neighbour_encoder(tokens, src_key_padding_mask=padding_mask)
        neighbour_summary = encoded[:, 0]
        focal = self.node_encoder(node)
        return self.output_head(torch.cat((focal, neighbour_summary), dim=1))


@dataclass
class InteractionTransformerRegressor:
    """Residual velocity predictor backed by masked neighbour self-attention."""

    network: _InteractionTransformer
    node_scaler: StandardScaler
    residual: bool = True
    seed: int = 20260722
    epochs_trained: int = 0
    best_epoch: int = 0
    training_history: tuple[float, ...] = ()
    validation_history: tuple[float, ...] = ()
    averaged_epochs: int = 1

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        """Predict world-frame velocity and enforce the requested speed cap."""

        if not np.isfinite(speed_cap) or speed_cap <= 0:
            raise ValueError("speed_cap must be positive and finite")
        node, edge, mask = _transformer_arrays(samples)
        device = next(self.network.parameters()).device
        self.network.eval()
        with torch.no_grad():
            local = self.network(
                torch.as_tensor(
                    self.node_scaler.transform(node),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(edge, dtype=torch.float32, device=device),
                torch.as_tensor(mask, dtype=torch.bool, device=device),
            ).cpu().numpy()
        if self.residual:
            local = local + local_target_from_vectors(samples, samples.features[:, 0])
        goal = samples.features[:, 1]
        lateral = np.column_stack((-goal[:, 1], goal[:, 0]))
        prediction = local[:, :1] * goal + local[:, 1:] * lateral
        speed = np.linalg.norm(prediction, axis=1)
        scale = np.minimum(1.0, speed_cap / np.maximum(speed, 1e-12))
        return prediction * scale[:, None]

    @property
    def parameter_count(self) -> int:
        """Number of trainable scalar parameters."""

        return int(sum(parameter.numel() for parameter in self.network.parameters()))


def _weighted_loss(
    network: _InteractionTransformer,
    node: torch.Tensor,
    edge: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    """Evaluate a run-balanced squared-vector loss without shuffling."""

    network.eval()
    total = 0.0
    mass = 0.0
    with torch.no_grad():
        for start in range(0, len(node), batch_size):
            stop = min(start + batch_size, len(node))
            prediction = network(node[start:stop], edge[start:stop], mask[start:stop])
            squared = torch.sum((prediction - target[start:stop]) ** 2, dim=1)
            current_weight = weights[start:stop]
            total += float(torch.sum(current_weight * squared))
            mass += float(torch.sum(current_weight))
    return total / max(mass, 1e-12)


def _average_states(
    states: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Average floating tensors and retain the last non-floating buffer."""

    if not states:
        raise ValueError("at least one state is required for weight averaging")
    averaged: dict[str, torch.Tensor] = {}
    for name in states[0]:
        tensors = [state[name] for state in states]
        if tensors[0].is_floating_point():
            mean = torch.stack([tensor.to(torch.float64) for tensor in tensors]).mean(dim=0)
            averaged[name] = mean.to(dtype=tensors[0].dtype)
        else:
            averaged[name] = tensors[-1].clone()
    return averaged


def fit_transformer_interaction_network(
    batches: Sequence[VelocitySamples],
    *,
    validation_batches: Sequence[VelocitySamples] | None = None,
    width: int = 32,
    num_heads: int = 4,
    num_layers: int = 1,
    feedforward_multiplier: int = 2,
    dropout: float = 0.0,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 512,
    seed: int = 20260722,
    residual: bool = True,
    early_stopping_patience: int | None = None,
    minimum_improvement: float = 0.0,
    weight_average_epochs: int = 0,
    num_threads: int | None = None,
) -> InteractionTransformerRegressor:
    """Fit a compact run-balanced interaction Transformer.

    ``validation_batches`` must contain complete runs and is never mixed into
    training.  When supplied, its run-balanced loss selects the checkpoint;
    otherwise training loss selects it.  ``early_stopping_patience`` counts
    consecutive epochs without at least ``minimum_improvement``.  If
    ``weight_average_epochs`` is positive, the final requested number of epoch
    states ending at the selected checkpoint are averaged.  This avoids
    averaging later epochs that triggered validation early stopping.
    """

    if epochs < 1:
        raise ValueError("epochs must be at least one")
    if batch_size < 1:
        raise ValueError("batch_size must be at least one")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be non-negative and finite")
    if early_stopping_patience is not None and early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least one or None")
    if not np.isfinite(minimum_improvement) or minimum_improvement < 0:
        raise ValueError("minimum_improvement must be non-negative and finite")
    if weight_average_epochs < 0:
        raise ValueError("weight_average_epochs must be non-negative")
    requested_threads = (
        min(4, max(1, torch.get_num_threads()))
        if num_threads is None
        else int(num_threads)
    )
    if requested_threads < 1:
        raise ValueError("num_threads must be at least one")

    train = _prepare_batches(batches, residual=residual)
    validation = (
        None
        if validation_batches is None
        else _prepare_batches(validation_batches, residual=residual)
    )
    scaler = StandardScaler().fit(train.node)

    train_node = torch.as_tensor(scaler.transform(train.node), dtype=torch.float32)
    train_edge = torch.as_tensor(train.edge, dtype=torch.float32)
    train_mask = torch.as_tensor(train.mask, dtype=torch.bool)
    train_target = torch.as_tensor(train.target, dtype=torch.float32)
    train_weights = torch.as_tensor(train.weights, dtype=torch.float32)
    dataset = TensorDataset(
        train_node,
        train_edge,
        train_mask,
        train_target,
        train_weights,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=False,
        num_workers=0,
    )

    validation_tensors: tuple[torch.Tensor, ...] | None = None
    if validation is not None:
        validation_tensors = (
            torch.as_tensor(scaler.transform(validation.node), dtype=torch.float32),
            torch.as_tensor(validation.edge, dtype=torch.float32),
            torch.as_tensor(validation.mask, dtype=torch.bool),
            torch.as_tensor(validation.target, dtype=torch.float32),
            torch.as_tensor(validation.weights, dtype=torch.float32),
        )

    previous_threads = torch.get_num_threads()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(requested_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        network = _InteractionTransformer(
            node_features=train.node.shape[1],
            edge_features=train.edge.shape[2],
            width=width,
            num_heads=num_heads,
            num_layers=num_layers,
            feedforward_multiplier=feedforward_multiplier,
            dropout=dropout,
        )
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        best_loss = float("inf")
        best_epoch = 0
        best_state = copy.deepcopy(network.state_dict())
        selected_average_states: list[dict[str, torch.Tensor]] = [best_state]
        recent_states: deque[dict[str, torch.Tensor]] = deque(
            maxlen=max(weight_average_epochs, 1)
        )
        stale_epochs = 0
        training_history: list[float] = []
        validation_history: list[float] = []

        for epoch in range(epochs):
            network.train()
            total = 0.0
            mass = 0.0
            for node_batch, edge_batch, mask_batch, target_batch, weight_batch in loader:
                optimizer.zero_grad()
                prediction = network(node_batch, edge_batch, mask_batch)
                squared = torch.sum((prediction - target_batch) ** 2, dim=1)
                loss = torch.sum(weight_batch * squared) / torch.sum(weight_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=5.0)
                optimizer.step()
                total += float(torch.sum(weight_batch * squared).detach())
                mass += float(torch.sum(weight_batch))
            training_loss = total / max(mass, 1e-12)
            training_history.append(training_loss)

            if validation_tensors is None:
                monitored_loss = training_loss
            else:
                monitored_loss = _weighted_loss(
                    network,
                    *validation_tensors,
                    batch_size=batch_size,
                )
                validation_history.append(monitored_loss)

            epoch_state = copy.deepcopy(network.state_dict())
            recent_states.append(epoch_state)
            if monitored_loss < best_loss - minimum_improvement:
                best_loss = monitored_loss
                best_epoch = epoch
                best_state = epoch_state
                selected_average_states = list(recent_states)
                stale_epochs = 0
            else:
                stale_epochs += 1
            if (
                early_stopping_patience is not None
                and stale_epochs >= early_stopping_patience
            ):
                break

        if weight_average_epochs > 0:
            network.load_state_dict(_average_states(selected_average_states))
            averaged_epochs = len(selected_average_states)
        else:
            network.load_state_dict(best_state)
            averaged_epochs = 1
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.set_num_threads(previous_threads)

    return InteractionTransformerRegressor(
        network=network,
        node_scaler=scaler,
        residual=residual,
        seed=seed,
        epochs_trained=len(training_history),
        best_epoch=best_epoch,
        training_history=tuple(training_history),
        validation_history=tuple(validation_history),
        averaged_epochs=averaged_epochs,
    )


# Short alias for experiment code; both names resolve to the identical fitter.
fit_interaction_transformer = fit_transformer_interaction_network


__all__ = [
    "InteractionTransformerRegressor",
    "fit_interaction_transformer",
    "fit_transformer_interaction_network",
]
