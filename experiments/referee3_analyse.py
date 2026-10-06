# Author: Amir Ghorbani
"""Analyse referee-3 member predictions without retraining or test tuning.

The analysis is intentionally forecast-level but all uncertainty summaries are
formed within complete runs before regime aggregation.  Crossing is labelled
development-informed.  The same code is later used once, unchanged, for the
locked T-junction evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, t, ttest_1samp
from sklearn.ensemble import HistGradientBoostingRegressor

from pedgeom.runtime_guard import (
    binary_detection_metrics,
    causal_delayed_two_expert_weights,
    causal_trailing_time_mean,
    ensemble_error_decomposition,
    finite_sample_upper_quantile,
    one_sided_gate_weights,
    per_run_tail_metrics,
    permute_weights_within_runs,
    risk_retention_curve,
    top_fraction_binary_labels,
)


@dataclass(frozen=True)
class Bundle:
    path: Path
    run: str
    role: str
    target: np.ndarray
    frame: np.ndarray
    origin_time_s: np.ndarray
    pedestrian_id: np.ndarray
    position: np.ndarray
    prior_velocity: np.ndarray
    neighbour_count: np.ndarray
    occupancy: np.ndarray
    members_raw: np.ndarray
    fallback_raw: np.ndarray
    constant_velocity: np.ndarray
    mahalanobis: np.ndarray


@dataclass(frozen=True)
class DrawCalibration:
    indices: tuple[int, ...]
    tau: float
    scale: float
    disagreement_tau: float
    disagreement_scale: float
    mahalanobis_tau: float
    mahalanobis_scale: float
    speed_tau: float
    speed_scale: float
    fixed_graph_weight: float


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _scalar(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def load_bundle(path: Path) -> Bundle:
    with np.load(path, allow_pickle=False) as data:
        return Bundle(
            path=path,
            run=_scalar(data["run"]),
            role=_scalar(data["role"]),
            target=np.array(data["target"], dtype=float),
            frame=np.array(data["frame"], dtype=np.int64),
            origin_time_s=np.array(data["origin_time_s"], dtype=float),
            pedestrian_id=np.array(data["pedestrian_id"]),
            position=np.array(data["position"], dtype=float),
            prior_velocity=np.array(data["prior_velocity"], dtype=float),
            neighbour_count=np.array(data["neighbour_count"], dtype=float),
            occupancy=np.array(data["occupancy"], dtype=float),
            members_raw=np.array(data["members_raw"], dtype=float),
            fallback_raw=np.array(data["fallback_raw"], dtype=float),
            constant_velocity=np.array(data["constant_velocity"], dtype=float),
            mahalanobis=np.array(data["mahalanobis"], dtype=float),
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_declared_bundle_path(project: Path, declared: str) -> Path:
    """Validate a manifest path while allowing the frozen cache junction.

    Large prediction archives may live behind the intentional
    ``data/processed/referee3_predictions`` junction.  A resolved-path check
    against the repository root alone therefore rejects valid archives on a
    different volume.  We first enforce lexical containment in that one
    declared project directory, then allow only resolved targets below the
    directory's own resolved root.  This admits the junction without admitting
    absolute paths, ``..`` traversal, sibling caches, or nested links that
    escape the allowlisted target.
    """

    relative = Path(declared)
    if relative.is_absolute():
        raise RuntimeError("training manifest bundle path must be relative")
    lexical_project = Path(os.path.abspath(project))
    lexical_root = Path(
        os.path.abspath(lexical_project / "data/processed/referee3_predictions")
    )
    lexical_path = Path(os.path.abspath(lexical_project / relative))
    if not lexical_path.is_relative_to(lexical_root):
        raise RuntimeError("training manifest bundle path leaves the prediction archive")
    try:
        resolved_root = lexical_root.resolve(strict=True)
        resolved_path = lexical_path.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(
            f"training manifest bundle path cannot be resolved: {lexical_path}"
        ) from error
    if not resolved_path.is_relative_to(resolved_root):
        raise RuntimeError(
            "training manifest bundle path leaves the allowlisted archive target"
        )
    return lexical_path


def _manifest_bundles(project: Path) -> tuple[list[Bundle], list[Bundle]]:
    """Load only checksum-verified archives declared by the training manifest."""

    manifest_path = project / "data/processed/referee3_training_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            "training manifest is missing; run referee3_generate_predictions.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = project / "configs/referee3_lock.json"
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("training manifest does not match the frozen referee-3 config")
    declared = manifest.get("bundles")
    if not isinstance(declared, list) or not declared:
        raise RuntimeError("training manifest contains no prediction bundles")
    oof: list[Bundle] = []
    tests: list[Bundle] = []
    seen: set[Path] = set()
    for row in declared:
        if not isinstance(row, dict) or "path" not in row or "sha256" not in row:
            raise RuntimeError("training manifest contains an invalid bundle entry")
        path = _resolve_declared_bundle_path(project, str(row["path"]))
        if path in seen:
            raise RuntimeError(f"duplicate bundle entry in manifest: {path.name}")
        seen.add(path)
        if not path.exists() or _sha256(path) != str(row["sha256"]):
            raise RuntimeError(f"prediction bundle checksum mismatch: {path}")
        bundle = load_bundle(path)
        if bundle.run != str(row.get("run")) or bundle.role != str(row.get("role")):
            raise RuntimeError(f"prediction bundle metadata mismatch: {path}")
        if path.name.startswith("oof_"):
            oof.append(bundle)
        elif path.name.startswith("test_"):
            tests.append(bundle)
        else:
            raise RuntimeError(f"unrecognised prediction bundle role: {path.name}")
    return oof, tests


def cap_vectors(vectors: np.ndarray, limit: float) -> np.ndarray:
    speed = np.linalg.norm(vectors, axis=-1)
    scale = np.minimum(1.0, limit / np.maximum(speed, 1e-12))
    return vectors * scale[..., None]


def selected_predictions(
    bundle: Bundle, indices: tuple[int, ...], speed_cap: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    members = cap_vectors(bundle.members_raw[np.asarray(indices)], speed_cap)
    graph = np.mean(members, axis=0)
    fallback = cap_vectors(bundle.fallback_raw, speed_cap)
    return members, graph, fallback


def _rolling(bundle: Bundle, values: np.ndarray, window_seconds: float) -> np.ndarray:
    # Explicit seconds make the calculation invariant to each archive's native
    # frame rate and to the 16 -> 10 Hz resampling used by the untouched test.
    return causal_trailing_time_mean(
        values,
        bundle.pedestrian_id,
        bundle.origin_time_s,
        frames_per_second=1.0,
        window_seconds=window_seconds,
    )


def signals(
    bundle: Bundle,
    members: np.ndarray,
    graph: np.ndarray,
    fallback: np.ndarray,
    window_seconds: float,
) -> dict[str, np.ndarray]:
    instantaneous_spread = np.sqrt(
        np.mean(np.sum((members - graph[None, :, :]) ** 2, axis=2), axis=0)
    )
    disagreement = np.linalg.norm(graph - fallback, axis=1)
    speed = np.linalg.norm(bundle.prior_velocity, axis=1)
    return {
        "spread": _rolling(bundle, instantaneous_spread, window_seconds),
        "disagreement": _rolling(bundle, disagreement, window_seconds),
        "mahalanobis": _rolling(bundle, bundle.mahalanobis, window_seconds),
        "speed": _rolling(bundle, speed, window_seconds),
    }


def _threshold_scale(values: np.ndarray) -> tuple[float, float]:
    first, third = np.quantile(values, [0.25, 0.75])
    return finite_sample_upper_quantile(values, 0.9), max(float(third - first), 1e-6)


def _run_balanced_rmse(
    targets: list[np.ndarray], predictions: list[np.ndarray]
) -> float:
    values = [
        np.sqrt(np.mean(np.sum((prediction - target) ** 2, axis=1)))
        for target, prediction in zip(targets, predictions, strict=True)
    ]
    return float(np.mean(values))


def _equal_run_sample_weights(parts: list[np.ndarray]) -> np.ndarray:
    """Give every complete run equal aggregate weight in row-wise fitting."""

    if not parts or any(len(part) == 0 for part in parts):
        raise ValueError("every fitted run must contain at least one row")
    return np.concatenate(
        [np.full(len(part), 1.0 / len(part), dtype=float) for part in parts]
    )


def _stable_run_seed(base_seed: int, run: str, replicate: int = 0) -> int:
    """Derive independent, reproducible random streams without Python hashing."""

    payload = f"{int(base_seed)}\0{run}\0{int(replicate)}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little")


def calibrate_draw(
    oof: list[Bundle],
    indices: tuple[int, ...],
    config: dict,
) -> DrawCalibration:
    score_parts: dict[str, list[np.ndarray]] = {
        "spread": [],
        "disagreement": [],
        "mahalanobis": [],
        "speed": [],
    }
    graphs: list[np.ndarray] = []
    fallbacks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    speed_cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    window = float(config["guard"]["trailing_window_seconds"])
    for bundle in oof:
        members, graph, fallback = selected_predictions(bundle, indices, speed_cap)
        current = signals(bundle, members, graph, fallback, window)
        for name in score_parts:
            score_parts[name].append(current[name])
        graphs.append(graph)
        fallbacks.append(fallback)
        targets.append(bundle.target)
    # The locked calibration envelope is a pooled-origin finite-sample
    # estimand.  We deliberately retain that convention here; complete-run
    # resampling is used later to quantify its uncertainty.  This differs from
    # learned row-wise comparators, whose optimisation weights each run equally.
    limits = {
        name: _threshold_scale(np.concatenate(parts))
        for name, parts in score_parts.items()
    }
    step = float(config["controls"]["fixed_blend_grid_step"])
    candidates = np.arange(0.0, 1.0 + step / 2.0, step)
    losses = [
        _run_balanced_rmse(
            targets,
            [
                weight * graph + (1.0 - weight) * fallback
                for graph, fallback in zip(graphs, fallbacks, strict=True)
            ],
        )
        for weight in candidates
    ]
    return DrawCalibration(
        indices=indices,
        tau=limits["spread"][0],
        scale=limits["spread"][1],
        disagreement_tau=limits["disagreement"][0],
        disagreement_scale=limits["disagreement"][1],
        mahalanobis_tau=limits["mahalanobis"][0],
        mahalanobis_scale=limits["mahalanobis"][1],
        speed_tau=limits["speed"][0],
        speed_scale=limits["speed"][1],
        fixed_graph_weight=float(candidates[int(np.argmin(losses))]),
    )


def draw_indices(
    pool_size: int, size: int, count: int, seed: int, *, include_primary: bool
) -> list[tuple[int, ...]]:
    if size > pool_size:
        raise ValueError(f"ensemble size {size} exceeds member pool {pool_size}")
    generator = np.random.default_rng(seed + 10_000 * size)
    output: list[tuple[int, ...]] = []
    if include_primary:
        output.append(tuple(range(size)))
    seen = set(output)
    while len(output) < count:
        candidate = tuple(sorted(generator.choice(pool_size, size=size, replace=False)))
        if candidate not in seen:
            seen.add(candidate)
            output.append(candidate)
    return output


def _prediction_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = np.linalg.norm(prediction - target, axis=1)
    squared = np.sum((prediction - target) ** 2, axis=1)
    tail = per_run_tail_metrics(error, np.zeros(len(error), dtype=int)).iloc[0]
    return {
        "rmse": float(np.sqrt(np.mean(squared))),
        "mae_vector": float(np.mean(error)),
        "q90": float(tail["q90"]),
        "q95": float(tail["q95"]),
        "q99": float(tail["q99"]),
        "cvar95": float(tail["cvar95"]),
        "cvar99": float(tail["cvar99"]),
        "large_error_rate": float(tail["large_error_rate"]),
    }


def _site_online_weights(
    bundle: Bundle,
    graph: np.ndarray,
    fallback: np.ndarray,
    *,
    learning_rate: float,
    fixed_share: float,
    label_delay_seconds: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    graph_loss = np.sum((graph - bundle.target) ** 2, axis=1)
    fallback_loss = np.sum((fallback - bundle.target) ** 2, axis=1)
    unique_frames = np.unique(bundle.origin_time_s)
    frame_loss = np.array(
        [
            [
                np.mean(graph_loss[bundle.origin_time_s == frame]),
                np.mean(fallback_loss[bundle.origin_time_s == frame]),
            ]
            for frame in unique_frames
        ]
    )
    result = causal_delayed_two_expert_weights(
        frame_loss,
        unique_frames,
        frames_per_second=1.0,
        label_delay_seconds=label_delay_seconds,
        learning_rate=learning_rate,
        fixed_share=fixed_share,
    )
    lookup = {frame: index for index, frame in enumerate(unique_frames)}
    index = np.array([lookup[frame] for frame in bundle.origin_time_s])
    return result.weights[index, 0], result.updates_applied[index]


def _pedestrian_online_weights(
    bundle: Bundle,
    graph: np.ndarray,
    fallback: np.ndarray,
    *,
    learning_rate: float,
    fixed_share: float,
    label_delay_seconds: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    losses = np.column_stack(
        (
            np.sum((graph - bundle.target) ** 2, axis=1),
            np.sum((fallback - bundle.target) ** 2, axis=1),
        )
    )
    result = causal_delayed_two_expert_weights(
        losses,
        bundle.origin_time_s,
        frames_per_second=1.0,
        label_delay_seconds=label_delay_seconds,
        learning_rate=learning_rate,
        fixed_share=fixed_share,
        stream_ids=bundle.pedestrian_id,
    )
    return result.weights[:, 0], result.updates_applied


def choose_online_rule(
    oof: list[Bundle], calibration: DrawCalibration, config: dict
) -> tuple[str, float, float, pd.DataFrame]:
    rows: list[dict] = []
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    label_delay = float(config["controls"]["online_experts"]["label_delay_seconds"])
    for scope in ("site", "pedestrian"):
        for learning_rate in config["controls"]["online_experts"]["learning_rates"]:
            for fixed_share in config["controls"]["online_experts"]["fixed_share_rates"]:
                run_values = []
                for bundle in oof:
                    _, graph, fallback = selected_predictions(bundle, calibration.indices, cap)
                    function = (
                        _site_online_weights if scope == "site" else _pedestrian_online_weights
                    )
                    weight, _ = function(
                        bundle,
                        graph,
                        fallback,
                        learning_rate=float(learning_rate),
                        fixed_share=float(fixed_share),
                        label_delay_seconds=label_delay,
                    )
                    prediction = weight[:, None] * graph + (1.0 - weight[:, None]) * fallback
                    run_values.append(_prediction_metrics(bundle.target, prediction)["rmse"])
                rows.append(
                    {
                        "scope": scope,
                        "learning_rate": float(learning_rate),
                        "fixed_share": float(fixed_share),
                        "oof_run_macro_rmse": float(np.mean(run_values)),
                    }
                )
    table = pd.DataFrame(rows).sort_values(
        ["oof_run_macro_rmse", "scope", "learning_rate", "fixed_share"]
    )
    best = table.iloc[0]
    return (
        str(best["scope"]),
        float(best["learning_rate"]),
        float(best["fixed_share"]),
        table,
    )


def _optimal_graph_weight(
    graph: np.ndarray, fallback: np.ndarray, target: np.ndarray
) -> np.ndarray:
    difference = graph - fallback
    denominator = np.sum(difference**2, axis=1)
    numerator = np.sum(difference * (target - fallback), axis=1)
    return np.clip(
        np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, 0.5),
            where=denominator > 1e-12,
        ),
        0.0,
        1.0,
    )


def fit_learned_gate(
    oof: list[Bundle], calibration: DrawCalibration, config: dict
) -> HistGradientBoostingRegressor:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    window = float(config["guard"]["trailing_window_seconds"])
    for bundle in oof:
        members, graph, fallback = selected_predictions(bundle, calibration.indices, cap)
        current = signals(bundle, members, graph, fallback, window)
        features.append(
            np.column_stack(
                (
                    current["spread"],
                    current["disagreement"],
                    current["mahalanobis"],
                    current["speed"],
                    bundle.neighbour_count,
                    bundle.occupancy,
                )
            )
        )
        targets.append(_optimal_graph_weight(graph, fallback, bundle.target))
    model = HistGradientBoostingRegressor(
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        learning_rate=0.05,
        l2_regularization=2.0,
        random_state=int(config["replication"]["member_seed_base"]),
    )
    return model.fit(
        np.concatenate(features),
        np.concatenate(targets),
        sample_weight=_equal_run_sample_weights(features),
    )


def _partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    ranked = [rankdata(value) for value in (x, y, control)]
    design = np.column_stack((np.ones(len(x)), ranked[2]))
    x_residual = ranked[0] - design @ np.linalg.lstsq(design, ranked[0], rcond=None)[0]
    y_residual = ranked[1] - design @ np.linalg.lstsq(design, ranked[1], rcond=None)[0]
    if np.ptp(x_residual) <= 1e-12 or np.ptp(y_residual) <= 1e-12:
        return float("nan")
    return float(pearsonr(x_residual, y_residual).statistic)


def _online_stream_age(bundle: Bundle, scope: str) -> np.ndarray:
    """Return causal age since the stream's first forecast origin."""

    if scope == "site":
        return bundle.origin_time_s - float(np.min(bundle.origin_time_s))
    if scope != "pedestrian":
        raise ValueError(f"unknown online scope: {scope}")
    first: dict[object, float] = {}
    age = np.empty(len(bundle.origin_time_s), dtype=float)
    for row, (pedestrian, time) in enumerate(
        zip(bundle.pedestrian_id, bundle.origin_time_s, strict=True)
    ):
        key = pedestrian.item() if hasattr(pedestrian, "item") else pedestrian
        first.setdefault(key, float(time))
        age[row] = float(time) - first[key]
    return age


def _evaluate_primary_draw(
    oof: list[Bundle],
    tests: list[Bundle],
    calibration: DrawCalibration,
    config: dict,
    output: Path,
) -> None:
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    window = float(config["guard"]["trailing_window_seconds"])
    learned_gate = fit_learned_gate(oof, calibration, config)
    online_scope, online_eta, online_share, online_grid = choose_online_rule(
        oof, calibration, config
    )
    online_grid.to_csv(output / "referee3_online_calibration.csv", index=False)
    metric_rows: list[dict] = []
    forecast_rows: list[pd.DataFrame] = []
    detection_rows: list[dict] = []
    risk_rows: list[dict] = []
    stratum_rows: list[dict] = []
    decomposition_rows: list[dict] = []
    targeting_rows: list[dict] = []
    correlation_rows: list[dict] = []
    firing_rows: list[dict] = []
    risk_curve_rows: list[dict] = []
    cold_start_rows: list[dict] = []
    activation_rows: list[dict] = []
    permutation_count = int(config["controls"]["within_run_gate_permutations"])
    permutation_seed = int(config["controls"]["permutation_seed"])
    online_delay = float(config["controls"]["online_experts"]["label_delay_seconds"])
    cold_start_seconds = float(config["controls"]["online_experts"]["cold_start_seconds"])
    retention_grid = np.linspace(0.01, 1.0, 100)

    for bundle in oof + tests:
        members, graph, fallback = selected_predictions(bundle, calibration.indices, cap)
        current = signals(bundle, members, graph, fallback, window)
        weights = {
            "GEL-Ped": one_sided_gate_weights(
                current["spread"],
                threshold=calibration.tau,
                scale=calibration.scale,
                kind=config["guard"]["primary_shape"],
            ),
            "disagreement guard": one_sided_gate_weights(
                current["disagreement"],
                threshold=calibration.disagreement_tau,
                scale=calibration.disagreement_scale,
            ),
            "Mahalanobis guard": one_sided_gate_weights(
                current["mahalanobis"],
                threshold=calibration.mahalanobis_tau,
                scale=calibration.mahalanobis_scale,
            ),
            "speed guard": one_sided_gate_weights(
                current["speed"],
                threshold=calibration.speed_tau,
                scale=calibration.speed_scale,
            ),
        }
        learned_features = np.column_stack(
            (
                current["spread"],
                current["disagreement"],
                current["mahalanobis"],
                current["speed"],
                bundle.neighbour_count,
                bundle.occupancy,
            )
        )
        weights["learned matched gate"] = np.clip(
            learned_gate.predict(learned_features), 0.0, 1.0
        )
        online_function = (
            _site_online_weights if online_scope == "site" else _pedestrian_online_weights
        )
        online_weight, online_updates = online_function(
            bundle,
            graph,
            fallback,
            learning_rate=online_eta,
            fixed_share=online_share,
            label_delay_seconds=online_delay,
        )
        weights["delayed-error online gate"] = online_weight
        predictions = {
            "constant velocity": cap_vectors(bundle.constant_velocity, cap),
            "one graph member": members[0],
            "graph ensemble": graph,
            "coordinate-wise member median": np.median(members, axis=0),
            "feature-based HGB fallback": fallback,
            "calibration-fixed blend": calibration.fixed_graph_weight * graph
            + (1.0 - calibration.fixed_graph_weight) * fallback,
        }
        for name, weight in weights.items():
            predictions[name] = weight[:, None] * graph + (1.0 - weight[:, None]) * fallback
        graph_error = np.linalg.norm(graph - bundle.target, axis=1)
        fallback_error = np.linalg.norm(fallback - bundle.target, axis=1)
        oracle = np.where(
            (graph_error <= fallback_error)[:, None], graph, fallback
        )
        predictions["future-label oracle"] = oracle
        mean_g = float(np.mean(weights["GEL-Ped"]))
        predictions["mean-g fixed diagnostic"] = mean_g * graph + (1.0 - mean_g) * fallback

        for name, prediction in predictions.items():
            metric_rows.append(
                {
                    "role": bundle.role,
                    "run": bundle.run,
                    "model": name,
                    "n": len(bundle.target),
                    "evaluation_status": (
                        "calibration-reused" if bundle.role == "calibration" else "locked-evaluation"
                    ),
                    **_prediction_metrics(bundle.target, prediction),
                }
            )

        online_age = _online_stream_age(bundle, online_scope)
        for phase, mask in {
            "cold_start": online_age < cold_start_seconds - 1e-12,
            "post_cold_start": online_age >= cold_start_seconds - 1e-12,
        }.items():
            if not np.any(mask):
                continue
            for model_name in (
                "graph ensemble",
                "GEL-Ped",
                "delayed-error online gate",
            ):
                cold_start_rows.append(
                    {
                        "role": bundle.role,
                        "run": bundle.run,
                        "online_scope": online_scope,
                        "phase": phase,
                        "model": model_name,
                        "n": int(np.sum(mask)),
                        **_prediction_metrics(
                            bundle.target[mask], predictions[model_name][mask]
                        ),
                    }
                )
        firing_rows.append(
            {
                "role": bundle.role,
                "run": bundle.run,
                "n": len(bundle.target),
                "fraction_spread_above_tau": float(
                    np.mean(current["spread"] > calibration.tau)
                ),
                "mean_graph_weight": mean_g,
                "q10_graph_weight": float(np.quantile(weights["GEL-Ped"], 0.1)),
                "tau": calibration.tau,
                "scale": calibration.scale,
            }
        )
        track_age = np.round(_online_stream_age(bundle, "pedestrian"), 9)
        for age in np.unique(track_age):
            mask = track_age == age
            activation_rows.append(
                {
                    "role": bundle.role,
                    "run": bundle.run,
                    "track_age_seconds": float(age),
                    "n": int(np.sum(mask)),
                    "mean_spread": float(np.mean(current["spread"][mask])),
                    "fraction_spread_above_tau": float(
                        np.mean(current["spread"][mask] > calibration.tau)
                    ),
                    "mean_graph_weight": float(np.mean(weights["GEL-Ped"][mask])),
                }
            )

        high_error = top_fraction_binary_labels(graph_error, 0.05)
        positive_errors = graph_error[high_error.astype(bool)]
        negative_errors = graph_error[~high_error.astype(bool)]
        for signal_name, signal in {
            **current,
            "neighbour density": bundle.neighbour_count / (np.pi * 3.0**2),
        }.items():
            values = binary_detection_metrics(high_error, signal)
            detection_rows.append(
                {
                    "role": bundle.role,
                    "run": bundle.run,
                    "signal": signal_name,
                    "positive_count": int(np.sum(high_error)),
                    "minimum_positive_error": float(np.min(positive_errors)),
                    "maximum_negative_error": float(np.max(negative_errors)),
                    **values,
                }
            )
            curve = risk_retention_curve(graph_error**2, signal)
            risk_rows.append(
                {
                    "role": bundle.role,
                    "run": bundle.run,
                    "signal": signal_name,
                    "aurc_squared_error": curve.aurc,
                    "curve_points": len(curve.retention),
                }
            )
            grid_indices = np.minimum(
                np.searchsorted(curve.retention, retention_grid, side="left"),
                len(curve.retention) - 1,
            )
            for requested, curve_index in zip(
                retention_grid, grid_indices, strict=True
            ):
                risk_curve_rows.append(
                    {
                        "role": bundle.role,
                        "run": bundle.run,
                        "signal": signal_name,
                        "requested_retention": float(requested),
                        "realized_retention": float(curve.retention[curve_index]),
                        "retained_squared_error_risk": float(curve.risk[curve_index]),
                    }
                )
        correlation_rows.append(
            {
                "role": bundle.role,
                "run": bundle.run,
                "partial_spearman_spread_error_given_speed": _partial_spearman(
                    current["spread"], graph_error, current["speed"]
                ),
            }
        )

        normalized = (current["spread"] - calibration.tau) / calibration.scale
        # The identity branch includes the threshold itself; right-closed bins
        # keep E == tau inside and reserve the final label for strictly >2 IQR.
        bins = np.digitize(normalized, [0.0, 1.0, 2.0], right=True)
        labels = ("inside", "0--1 IQR", "1--2 IQR", ">2 IQR")
        stratum_predictions = {
            "graph ensemble": graph,
            "GEL-Ped": predictions["GEL-Ped"],
            "feature-based HGB fallback": fallback,
            "calibration-fixed blend": predictions["calibration-fixed blend"],
            "mean-g fixed diagnostic": predictions["mean-g fixed diagnostic"],
            "future-label oracle": oracle,
        }
        permutation_by_stratum: dict[int, list[float]] = {
            code: [] for code in range(len(labels))
        }
        if bundle.role != "calibration":
            observed_masks = [("all", np.ones(len(bundle.target), dtype=bool))]
            observed_masks.extend(
                (label, bins == code) for code, label in enumerate(labels)
            )
            observed_by_stratum = {
                label: float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (
                                    predictions["GEL-Ped"][mask]
                                    - bundle.target[mask]
                                )
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                )
                for label, mask in observed_masks
                if np.any(mask)
            }
            for permutation in range(permutation_count):
                permuted = permute_weights_within_runs(
                    weights["GEL-Ped"],
                    np.repeat(bundle.run, len(bundle.target)),
                    seed=_stable_run_seed(permutation_seed, bundle.run, permutation),
                )
                prediction = (
                    permuted[:, None] * graph
                    + (1.0 - permuted[:, None]) * fallback
                )
                for label, mask in observed_masks:
                    if not np.any(mask):
                        continue
                    permuted_rmse = float(
                        np.sqrt(
                            np.mean(
                                np.sum(
                                    (prediction[mask] - bundle.target[mask]) ** 2,
                                    axis=1,
                                )
                            )
                        )
                    )
                    targeting_rows.append(
                        {
                            "role": bundle.role,
                            "run": bundle.run,
                            "stratum": label,
                            "n": int(np.sum(mask)),
                            "permutation": permutation,
                            "observed_gel_rmse": observed_by_stratum[label],
                            "permuted_rmse": permuted_rmse,
                        }
                    )
                    if label != "all":
                        permutation_by_stratum[labels.index(label)].append(
                            permuted_rmse
                        )
        for code, label in enumerate(labels):
            mask = bins == code
            if not np.any(mask):
                continue
            for name, prediction in stratum_predictions.items():
                stratum_rows.append(
                    {
                        "role": bundle.role,
                        "run": bundle.run,
                        "stratum": label,
                        "model": name,
                        "n": int(np.sum(mask)),
                        "share": float(np.mean(mask)),
                        "rmse": float(
                            np.sqrt(
                                np.mean(
                                    np.sum(
                                        (prediction[mask] - bundle.target[mask]) ** 2,
                                        axis=1,
                                    )
                                )
                            )
                        ),
                    }
                )
            if permutation_by_stratum[code]:
                permutation_values = np.asarray(
                    permutation_by_stratum[code], dtype=float
                )
                stratum_rows.append(
                    {
                        "role": bundle.role,
                        "run": bundle.run,
                        "stratum": label,
                        "model": "permuted-g diagnostic",
                        "n": int(np.sum(mask)),
                        "share": float(np.mean(mask)),
                        "rmse": float(np.mean(permutation_values)),
                        "rmse_q025": float(np.quantile(permutation_values, 0.025)),
                        "rmse_q975": float(np.quantile(permutation_values, 0.975)),
                        "n_permutations": len(permutation_values),
                    }
                )
        decomposition = ensemble_error_decomposition(members, bundle.target)
        for code, label in enumerate(labels):
            mask = bins == code
            if not np.any(mask):
                continue
            decomposition_rows.append(
                {
                    "role": bundle.role,
                    "run": bundle.run,
                    "stratum": label,
                    "n": int(np.sum(mask)),
                    "shared_mse": float(
                        np.mean(decomposition.shared_squared_error[mask])
                    ),
                    "between_member_variance": float(
                        np.mean(decomposition.between_member_variance[mask])
                    ),
                    "member_mse": float(
                        np.mean(decomposition.mean_member_squared_error[mask])
                    ),
                    "identity_residual": float(
                        np.mean(decomposition.mean_member_squared_error[mask])
                        - np.mean(decomposition.shared_squared_error[mask])
                        - np.mean(decomposition.between_member_variance[mask])
                    ),
                }
            )

        if bundle.role != "calibration":
            forecast_rows.append(
                pd.DataFrame(
                    {
                        "role": bundle.role,
                        "run": bundle.run,
                        "frame": bundle.frame,
                        "origin_time_s": bundle.origin_time_s,
                        "pedestrian_id": bundle.pedestrian_id,
                        "target_x": bundle.target[:, 0],
                        "target_y": bundle.target[:, 1],
                        "graph_x": graph[:, 0],
                        "graph_y": graph[:, 1],
                        "fallback_x": fallback[:, 0],
                        "fallback_y": fallback[:, 1],
                        "gel_x": predictions["GEL-Ped"][:, 0],
                        "gel_y": predictions["GEL-Ped"][:, 1],
                        "spread": current["spread"],
                        "graph_weight": weights["GEL-Ped"],
                        "mahalanobis": current["mahalanobis"],
                        "graph_fallback_disagreement": current["disagreement"],
                        "speed_mps": current["speed"],
                        "neighbour_count": bundle.neighbour_count,
                        "occupancy": bundle.occupancy,
                        "online_updates": online_updates,
                    }
                )
            )

    pd.DataFrame(metric_rows).to_csv(output / "referee3_primary_run_metrics.csv", index=False)
    pd.DataFrame(detection_rows).to_csv(output / "referee3_detection_metrics.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(output / "referee3_risk_retention.csv", index=False)
    pd.DataFrame(risk_curve_rows).to_csv(
        output / "referee3_risk_retention_curve.csv", index=False
    )
    pd.DataFrame(stratum_rows).to_csv(output / "referee3_targeting_by_stratum.csv", index=False)
    pd.DataFrame(decomposition_rows).to_csv(output / "referee3_error_decomposition.csv", index=False)
    pd.DataFrame(targeting_rows).to_csv(output / "referee3_permuted_gate.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(output / "referee3_speed_partial_correlation.csv", index=False)
    pd.DataFrame(firing_rows).to_csv(output / "referee3_firing_rates.csv", index=False)
    pd.DataFrame(cold_start_rows).to_csv(
        output / "referee3_online_cold_start_metrics.csv", index=False
    )
    pd.DataFrame(activation_rows).to_csv(
        output / "referee3_activation_by_track_age.csv", index=False
    )
    pd.concat(forecast_rows, ignore_index=True).to_csv(
        output / "referee3_forecasts_primary.csv.gz", index=False, compression="gzip"
    )
    (output / "referee3_online_selection.json").write_text(
        json.dumps(
            {
                "scope": online_scope,
                "learning_rate": online_eta,
                "fixed_share": online_share,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _draw_analysis(
    oof: list[Bundle], tests: list[Bundle], config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool_size = min(bundle.members_raw.shape[0] for bundle in oof + tests)
    seed = int(config["replication"]["ensemble_draw_seed"])
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    window = float(config["guard"]["trailing_window_seconds"])
    metric_rows: list[dict] = []
    calibration_rows: list[dict] = []
    for size in config["replication"]["ensemble_sizes"]:
        count = int(config["replication"]["draws_per_size"])
        draws = draw_indices(
            pool_size,
            int(size),
            count,
            seed,
            include_primary=int(size) == int(config["replication"]["primary_ensemble_size"]),
        )
        for draw_index, indices in enumerate(draws):
            calibration = calibrate_draw(oof, indices, config)
            calibration_rows.append(
                {
                    "ensemble_size": size,
                    "draw": draw_index,
                    "member_indices": ";".join(map(str, indices)),
                    "tau": calibration.tau,
                    "scale": calibration.scale,
                    "fixed_graph_weight": calibration.fixed_graph_weight,
                }
            )
            for bundle in tests:
                members, graph, fallback = selected_predictions(bundle, indices, cap)
                current = signals(bundle, members, graph, fallback, window)
                weight = one_sided_gate_weights(
                    current["spread"],
                    threshold=calibration.tau,
                    scale=calibration.scale,
                    kind=config["guard"]["primary_shape"],
                )
                guard = weight[:, None] * graph + (1.0 - weight[:, None]) * fallback
                graph_metrics = _prediction_metrics(bundle.target, graph)
                guard_metrics = _prediction_metrics(bundle.target, guard)
                decomposition = ensemble_error_decomposition(members, bundle.target)
                member_rmse = np.sqrt(
                    np.mean(
                        np.sum((members - bundle.target[None, :, :]) ** 2, axis=2),
                        axis=1,
                    )
                )
                median = float(np.median(member_rmse))
                mad = float(np.median(np.abs(member_rmse - median)))
                scaled_mad = 1.4826 * mad
                divergent = int(np.sum(member_rmse > median + 3.0 * max(scaled_mad, 1e-12)))
                row = {
                    "role": bundle.role,
                    "run": bundle.run,
                    "ensemble_size": size,
                    "draw": draw_index,
                    "member_indices": ";".join(map(str, indices)),
                    "mean_graph_weight": float(np.mean(weight)),
                    "firing_rate": float(np.mean(current["spread"] > calibration.tau)),
                    "divergent_members": divergent,
                    "shared_mse": decomposition.shared_mse,
                    "between_member_variance": (
                        decomposition.between_member_variance_mean
                    ),
                    "member_mse": decomposition.member_mse,
                    "decomposition_identity_residual": (
                        decomposition.member_mse
                        - decomposition.shared_mse
                        - decomposition.between_member_variance_mean
                    ),
                }
                for endpoint in (
                    "rmse",
                    "q90",
                    "q95",
                    "q99",
                    "cvar95",
                    "cvar99",
                    "large_error_rate",
                ):
                    row[f"graph_{endpoint}"] = graph_metrics[endpoint]
                    row[f"guard_{endpoint}"] = guard_metrics[endpoint]
                    row[f"guard_minus_graph_{endpoint}"] = (
                        guard_metrics[endpoint] - graph_metrics[endpoint]
                    )
                metric_rows.append(row)
            print(f"analysed M={size}, draw {draw_index + 1}/{len(draws)}", flush=True)
    return pd.DataFrame(metric_rows), pd.DataFrame(calibration_rows)


def _crossed_draw_run_inference(
    metrics: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    """Summarise effects while preserving run and member-draw replication axes.

    For each role, ensemble size, and endpoint, the complete crossed run-by-draw
    matrix is aggregated.  The interval resamples complete runs and complete
    member draws independently and takes their Cartesian submatrix.  It is a
    transparent two-axis sensitivity interval, not a claim that overlapping
    member subsets are statistically independent training experiments.
    """

    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    endpoints = (
        "rmse",
        "q90",
        "q95",
        "q99",
        "cvar95",
        "cvar99",
        "large_error_rate",
    )
    rows: list[dict] = []
    grouped = metrics.groupby(["role", "ensemble_size"], sort=True)
    for (role, ensemble_size), group in grouped:
        for endpoint in endpoints:
            column = f"guard_minus_graph_{endpoint}"
            matrix_frame = group.pivot(index="run", columns="draw", values=column)
            if matrix_frame.isna().any().any():
                raise ValueError(
                    f"incomplete run-by-draw matrix for {role}, M={ensemble_size}, {endpoint}"
                )
            matrix = matrix_frame.to_numpy(float)
            n_runs, n_draws = matrix.shape
            generator = np.random.default_rng(
                _stable_run_seed(
                    seed, f"{role}|{int(ensemble_size)}|{endpoint}"
                )
            )
            bootstrap = np.empty(repetitions, dtype=float)
            for replicate in range(repetitions):
                run_indices = generator.integers(0, n_runs, size=n_runs)
                draw_indices_selected = generator.integers(
                    0, n_draws, size=n_draws
                )
                bootstrap[replicate] = float(
                    np.mean(matrix[np.ix_(run_indices, draw_indices_selected)])
                )
            run_macro_by_draw = np.mean(matrix, axis=0)
            draw_macro_by_run = np.mean(matrix, axis=1)
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            rows.append(
                {
                    "role": role,
                    "ensemble_size": int(ensemble_size),
                    "endpoint": endpoint,
                    "n_runs": n_runs,
                    "n_draws": n_draws,
                    "mean_guard_minus_graph": float(np.mean(matrix)),
                    "draw_macro_median": float(np.median(run_macro_by_draw)),
                    "draw_macro_q025": float(
                        np.quantile(run_macro_by_draw, 0.025)
                    ),
                    "draw_macro_q975": float(
                        np.quantile(run_macro_by_draw, 0.975)
                    ),
                    "fraction_draws_with_improvement": float(
                        np.mean(run_macro_by_draw < 0.0)
                    ),
                    "between_draw_sd": float(
                        np.std(run_macro_by_draw, ddof=1)
                        if n_draws > 1
                        else 0.0
                    ),
                    "between_run_sd": float(
                        np.std(draw_macro_by_run, ddof=1) if n_runs > 1 else 0.0
                    ),
                    "crossed_bootstrap_q025": float(low),
                    "crossed_bootstrap_q975": float(high),
                    "bootstrap_repetitions": repetitions,
                    "bootstrap_unit": "complete runs crossed with member draws",
                }
            )
    return pd.DataFrame(rows)


def _calibration_bootstrap(
    oof: list[Bundle], calibration: DrawCalibration, config: dict
) -> pd.DataFrame:
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    window = float(config["guard"]["trailing_window_seconds"])
    by_run = []
    for bundle in oof:
        members, graph, fallback = selected_predictions(bundle, calibration.indices, cap)
        by_run.append(signals(bundle, members, graph, fallback, window)["spread"])
    generator = np.random.default_rng(int(config["endpoints"]["bootstrap_seed"]))
    rows = []
    for replicate in range(int(config["endpoints"]["bootstrap_replicates"])):
        selected = generator.integers(0, len(by_run), size=len(by_run))
        values = np.concatenate([by_run[index] for index in selected])
        tau, scale = _threshold_scale(values)
        rows.append({"replicate": replicate, "tau": tau, "scale": scale})
    return pd.DataFrame(rows)


def _calibration_bootstrap_summary(
    bootstrap: pd.DataFrame, calibration: DrawCalibration
) -> pd.DataFrame:
    rows = []
    for quantity, estimate in (
        ("tau", calibration.tau),
        ("scale", calibration.scale),
    ):
        low, high = np.quantile(bootstrap[quantity].to_numpy(float), [0.025, 0.975])
        rows.append(
            {
                "quantity": quantity,
                "pooled_origin_estimate": estimate,
                "complete_run_bootstrap_mean": float(bootstrap[quantity].mean()),
                "complete_run_bootstrap_q025": float(low),
                "complete_run_bootstrap_q975": float(high),
                "bootstrap_replicates": len(bootstrap),
            }
        )
    return pd.DataFrame(rows)


def _window_shape_sensitivity(
    oof: list[Bundle], tests: list[Bundle], indices: tuple[int, ...], config: dict
) -> pd.DataFrame:
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    rows: list[dict] = []
    windows = {1: 0.0, 3: 0.8, 5: 1.6, 10: 3.6}
    for origins, window in windows.items():
        score_parts = []
        for bundle in oof:
            members, graph, fallback = selected_predictions(bundle, indices, cap)
            score_parts.append(signals(bundle, members, graph, fallback, window)["spread"])
        tau, scale = _threshold_scale(np.concatenate(score_parts))
        for shape in config["guard"]["shape_sensitivities"] + [
            config["guard"]["primary_shape"]
        ]:
            for bundle in tests:
                members, graph, fallback = selected_predictions(bundle, indices, cap)
                score = signals(bundle, members, graph, fallback, window)["spread"]
                weight = one_sided_gate_weights(
                    score, threshold=tau, scale=scale, kind=shape
                )
                prediction = weight[:, None] * graph + (1.0 - weight[:, None]) * fallback
                rows.append(
                    {
                        "role": bundle.role,
                        "run": bundle.run,
                        "origins": origins,
                        "window_seconds": window,
                        "shape": shape,
                        "tau": tau,
                        "scale": scale,
                        "rmse": _prediction_metrics(bundle.target, prediction)["rmse"],
                        "mean_graph_weight": float(np.mean(weight)),
                    }
                )
    return pd.DataFrame(rows)


def _tost_row(values: np.ndarray, margin: float) -> dict[str, float | bool | int]:
    """Return a paired run-level TOST and its corresponding 90% interval."""

    difference = np.asarray(values, dtype=float)
    difference = difference[np.isfinite(difference)]
    n = len(difference)
    if n < 2:
        return {
            "n_runs": n,
            "mean_difference": float(np.mean(difference)) if n else float("nan"),
            "margin": margin,
            "ci90_lower": float("nan"),
            "ci90_upper": float("nan"),
            "p_lower": float("nan"),
            "p_upper": float("nan"),
            "tost_p": float("nan"),
            "equivalent_at_0.05": False,
        }
    lower_test = ttest_1samp(difference, -margin, alternative="greater")
    upper_test = ttest_1samp(difference, margin, alternative="less")
    mean = float(np.mean(difference))
    standard_error = float(np.std(difference, ddof=1) / np.sqrt(n))
    half_width = float(t.ppf(0.95, n - 1) * standard_error)
    tost_p = float(max(lower_test.pvalue, upper_test.pvalue))
    return {
        "n_runs": n,
        "mean_difference": mean,
        "margin": margin,
        "ci90_lower": mean - half_width,
        "ci90_upper": mean + half_width,
        "p_lower": float(lower_test.pvalue),
        "p_upper": float(upper_test.pvalue),
        "tost_p": tost_p,
        "equivalent_at_0.05": bool(tost_p < 0.05),
    }


def _equivalence(
    metrics: pd.DataFrame,
    *,
    absolute_margin: float,
    relative_margin: float,
    primary_ensemble_size: int,
) -> pd.DataFrame:
    """Test both frozen absolute and relative paired run-level margins."""

    rows = []
    primary = metrics[
        metrics["ensemble_size"].eq(primary_ensemble_size)
        & metrics["draw"].eq(0)
    ]
    for role, group in primary.groupby("role", sort=False):
        absolute = group["guard_minus_graph_rmse"].to_numpy(float)
        graph = group["graph_rmse"].to_numpy(float)
        relative = np.divide(
            absolute,
            graph,
            out=np.full_like(absolute, np.nan),
            where=np.abs(graph) > 1e-12,
        )
        rows.append(
            {
                "role": role,
                "margin_type": "absolute_mps",
                **_tost_row(absolute, absolute_margin),
            }
        )
        rows.append(
            {
                "role": role,
                "margin_type": "relative_to_run_graph_rmse",
                **_tost_row(relative, relative_margin),
            }
        )
    return pd.DataFrame(rows)


def analyse(args: argparse.Namespace) -> None:
    project = project_root()
    config = json.loads((project / "configs/referee3_lock.json").read_text())
    output = project / "data/processed"
    oof, tests = _manifest_bundles(project)
    if len(oof) != 7 or not tests:
        raise RuntimeError("prediction bundles are incomplete; run referee3_generate_predictions.py")
    pool_size = min(bundle.members_raw.shape[0] for bundle in oof + tests)
    required = max(config["replication"]["ensemble_sizes"])
    if pool_size < required:
        raise RuntimeError(f"member pool {pool_size} is smaller than required M={required}")
    primary = tuple(int(value) for value in config["replication"]["primary_member_indices"])
    calibration = calibrate_draw(oof, primary, config)
    _evaluate_primary_draw(oof, tests, calibration, config, output)
    draw_metrics, draw_calibration = _draw_analysis(oof, tests, config)
    draw_metrics.to_csv(output / "referee3_draw_run_metrics.csv", index=False)
    draw_calibration.to_csv(output / "referee3_draw_calibration.csv", index=False)
    _crossed_draw_run_inference(
        draw_metrics,
        repetitions=int(config["endpoints"]["bootstrap_replicates"]),
        seed=int(config["endpoints"]["bootstrap_seed"]),
    ).to_csv(output / "referee3_draw_run_inference.csv", index=False)
    calibration_bootstrap = _calibration_bootstrap(oof, calibration, config)
    calibration_bootstrap.to_csv(
        output / "referee3_calibration_bootstrap.csv", index=False
    )
    _calibration_bootstrap_summary(calibration_bootstrap, calibration).to_csv(
        output / "referee3_calibration_bootstrap_summary.csv", index=False
    )
    _window_shape_sensitivity(oof, tests, primary, config).to_csv(
        output / "referee3_window_shape_sensitivity.csv", index=False
    )
    _equivalence(
        draw_metrics,
        absolute_margin=float(
            config["endpoints"]["equivalence_absolute_margin_mps"]
        ),
        relative_margin=float(config["endpoints"]["equivalence_relative_margin"]),
        primary_ensemble_size=int(config["replication"]["primary_ensemble_size"]),
    ).to_csv(output / "referee3_equivalence.csv", index=False)
    (output / "referee3_primary_calibration.json").write_text(
        json.dumps(
            {
                **calibration.__dict__,
                "calibration_weighting": "pooled eligible origins across complete OOF runs",
                "uncertainty_resampling_unit": "complete OOF run",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("referee-3 analysis complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    analyse(parser.parse_args())


if __name__ == "__main__":
    main()
