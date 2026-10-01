# Author: Amir Ghorbani
"""Protocol-matched graph interaction network for pedestrian velocity prediction.

The model is deliberately compact.  A shared edge network encodes each nearby
pedestrian, learned attention pools the unordered messages, and a node head
predicts a correction to the observed velocity.  It is used as a strong neural
interaction baseline under exactly the same information and 0.4-s target as
GEL-Ped.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy

import numpy as np
from sklearn.preprocessing import StandardScaler

from pedgeom.benchmarks import invariant_design, local_target, local_target_from_vectors
from pedgeom.calibration import VelocitySamples

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:  # pragma: no cover - gives a useful optional-dependency error
    raise ImportError(
        "The graph benchmark requires PyTorch; install the CPU build listed in "
        "requirements-graph.txt."
    ) from error


def _graph_arrays(samples: VelocitySamples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized-ready node inputs, local edge inputs, and masks."""

    if samples.raw_neighbours is None:
        raise ValueError("raw-neighbour states are required for the graph benchmark")
    goal = samples.features[:, 1]
    lateral = np.column_stack((-goal[:, 1], goal[:, 0]))
    raw = samples.raw_neighbours
    position = raw[:, :, :2]
    velocity = raw[:, :, 2:4]
    mask = raw[:, :, 4] > 0.5
    position_local = np.stack(
        (
            np.einsum("nkc,nc->nk", position, goal),
            np.einsum("nkc,nc->nk", position, lateral),
        ),
        axis=2,
    )
    velocity_local = np.stack(
        (
            np.einsum("nkc,nc->nk", velocity, goal),
            np.einsum("nkc,nc->nk", velocity, lateral),
        ),
        axis=2,
    )
    distance = np.linalg.norm(position, axis=2)
    closing = np.maximum(
        -np.divide(
            np.sum(position * velocity, axis=2),
            distance,
            out=np.zeros_like(distance),
            where=distance > 1e-8,
        ),
        0.0,
    )
    edge = np.concatenate(
        (
            position_local / 3.0,
            velocity_local / 3.0,
            (distance / 3.0)[:, :, None],
            (closing / 3.0)[:, :, None],
            mask[:, :, None].astype(float),
        ),
        axis=2,
    )
    edge[~mask] = 0.0
    return invariant_design(samples), edge, mask


class _InteractionGraph(nn.Module):
    def __init__(self, node_features: int = 12, width: int = 48) -> None:
        super().__init__()
        self.edge_encoder = nn.Sequential(
            nn.Linear(7, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )
        self.edge_attention = nn.Linear(width, 1)
        self.node_head = nn.Sequential(
            nn.Linear(node_features + width, 2 * width),
            nn.ReLU(),
            nn.Linear(2 * width, width),
            nn.ReLU(),
            nn.Linear(width, 2),
        )

    def forward(
        self, node: torch.Tensor, edge: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        message = self.edge_encoder(edge)
        logits = self.edge_attention(message).squeeze(-1)
        logits = logits.masked_fill(~mask, -1e9)
        attention = torch.softmax(logits, dim=1) * mask.float()
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.sum(attention[:, :, None] * message, dim=1)
        return self.node_head(torch.cat((node, pooled), dim=1))


@dataclass
class GraphInteractionRegressor:
    """Permutation-invariant neural message passing around a velocity anchor."""

    network: _InteractionGraph
    node_scaler: StandardScaler
    residual: bool = True

    def predict(self, samples: VelocitySamples, speed_cap: float = 2.5) -> np.ndarray:
        node, edge, mask = _graph_arrays(samples)
        device = next(self.network.parameters()).device
        self.network.eval()
        with torch.no_grad():
            residual = self.network(
                torch.as_tensor(
                    self.node_scaler.transform(node), dtype=torch.float32, device=device
                ),
                torch.as_tensor(edge, dtype=torch.float32, device=device),
                torch.as_tensor(mask, dtype=torch.bool, device=device),
            ).cpu().numpy()
        local = residual
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
        return int(sum(parameter.numel() for parameter in self.network.parameters()))


def fit_graph_interaction_network(
    batches: list[VelocitySamples],
    *,
    width: int = 48,
    epochs: int = 45,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 512,
    seed: int = 20260722,
    residual: bool = True,
) -> GraphInteractionRegressor:
    """Fit the graph baseline with equal aggregate weight for every run."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    node_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    for batch in batches:
        node, edge, mask = _graph_arrays(batch)
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
    weights *= len(weights) / weights.sum()
    scaler = StandardScaler().fit(node)
    dataset = TensorDataset(
        torch.as_tensor(scaler.transform(node), dtype=torch.float32),
        torch.as_tensor(edge, dtype=torch.float32),
        torch.as_tensor(mask, dtype=torch.bool),
        torch.as_tensor(target, dtype=torch.float32),
        torch.as_tensor(weights, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False
    )
    network = _InteractionGraph(node_features=node.shape[1], width=width)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_state = copy.deepcopy(network.state_dict())
    best_loss = float("inf")
    for _ in range(epochs):
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
        epoch_loss = total / max(mass, 1e-12)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(network.state_dict())
    network.load_state_dict(best_state)
    return GraphInteractionRegressor(
        network=network, node_scaler=scaler, residual=residual
    )
