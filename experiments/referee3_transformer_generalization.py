# Author: Amir Ghorbani
"""Architecture-generality audit with a compact interaction Transformer.

The script uses the locked referee-3 Juelich construction and never overwrites
the graph predictions or frozen referee-2 evidence.  Five deterministic
Transformer members are calibrated by leave-one-complete-run-out (LORO)
predictions.  Within every fit, a different complete training run is used for
early stopping; the selected epoch count is then refit on all runs available
to that LORO fold.  The deployed ensemble is treated identically.

The guard is the locked 1.6-s causal mean of member disagreement, with a
finite-sample 90% threshold and calibration IQR scale.  Transformer and the
feature-based HGB fallback use the same prediction speed cap.  Outputs are
cached exclusively under ``referee3_transformer_*`` names.

Run ``python experiments/referee3_transformer_generalization.py --dry-run``
for a small synthetic workflow check.  The real experiment is intentionally
not launched by this module's tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from causal_anchor_revision import anchor, settings
from pedgeom.benchmarks import fit_gradient_boosted_residual
from pedgeom.calibration import FEATURE_NAMES, VelocitySamples
from pedgeom.runtime_guard import (
    causal_trailing_time_mean,
    empirical_upper_cvar,
    finite_sample_upper_quantile,
    one_sided_gate_weights,
)
from pedgeom.transformer_baseline import fit_transformer_interaction_network
from referee3_generate_predictions import (
    _atomic_savez,
    _atomic_write_json,
    _batch_fingerprint,
    _member_seeds,
    _sha256,
    build_julich_batches,
    project_root,
)


_CACHE_SCHEMA = 1
_OUTPUT_SCHEMA = 1


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "width": int(args.width),
        "num_heads": int(args.num_heads),
        "num_layers": int(args.num_layers),
        "max_epochs": int(args.max_epochs),
        "patience": int(args.patience),
        "weight_average_epochs": int(args.weight_average_epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "num_threads": int(args.num_threads),
        "residual": True,
        "wall_features": False,
    }


def _cache_signature(
    project: Path,
    training: Sequence[VelocitySamples],
    validation_index: int,
    evaluation: Sequence[VelocitySamples],
    seed: int,
    options: dict[str, Any],
    speed_cap: float,
) -> str:
    payload = {
        "schema": _CACHE_SCHEMA,
        "source_sha256": _sha256(project / "src/pedgeom/transformer_baseline.py"),
        "training": [_batch_fingerprint(batch) for batch in training],
        "validation_index": validation_index,
        "evaluation": [_batch_fingerprint(batch) for batch in evaluation],
        "seed": seed,
        "options": options,
        "speed_cap": speed_cap,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_member_cache(
    path: Path,
    *,
    signature: str,
    evaluation: Sequence[VelocitySamples],
) -> tuple[list[np.ndarray], dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["cache_schema"].item()) != _CACHE_SCHEMA:
                return None
            if str(payload["signature"].item()) != signature:
                return None
            expected_names = np.asarray([batch.run for batch in evaluation])
            expected_lengths = np.asarray([len(batch.target) for batch in evaluation])
            if not np.array_equal(payload["run_names"], expected_names):
                return None
            if not np.array_equal(payload["run_lengths"], expected_lengths):
                return None
            predictions = []
            for index, batch in enumerate(evaluation):
                prediction = np.asarray(payload[f"prediction_{index}"], dtype=float)
                if prediction.shape != (len(batch.target), 2):
                    return None
                if not np.all(np.isfinite(prediction)):
                    return None
                predictions.append(prediction)
            metadata = {
                "selected_epochs": int(payload["selected_epochs"].item()),
                "tuning_epochs": int(payload["tuning_epochs"].item()),
                "averaged_epochs": int(payload["averaged_epochs"].item()),
                "parameter_count": int(payload["parameter_count"].item()),
                "seconds": float(payload["seconds"].item()),
            }
    except (KeyError, OSError, ValueError, EOFError):
        return None
    return predictions, metadata


def _fit_validated_member(
    training: Sequence[VelocitySamples],
    *,
    validation_index: int,
    evaluation: Sequence[VelocitySamples],
    seed: int,
    options: dict[str, Any],
    speed_cap: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Select epochs on one whole run, then refit on every available run."""

    if len(training) < 2:
        raise ValueError("validated Transformer fitting requires at least two runs")
    if not 0 <= validation_index < len(training):
        raise ValueError("validation_index is outside the training-run list")
    validation = training[validation_index]
    tuning_training = [
        batch for index, batch in enumerate(training) if index != validation_index
    ]
    started = time.perf_counter()
    tuned = fit_transformer_interaction_network(
        tuning_training,
        validation_batches=[validation],
        width=int(options["width"]),
        num_heads=int(options["num_heads"]),
        num_layers=int(options["num_layers"]),
        epochs=int(options["max_epochs"]),
        learning_rate=float(options["learning_rate"]),
        weight_decay=float(options["weight_decay"]),
        batch_size=int(options["batch_size"]),
        seed=seed,
        residual=True,
        early_stopping_patience=int(options["patience"]),
        weight_average_epochs=int(options["weight_average_epochs"]),
        num_threads=int(options["num_threads"]),
    )
    selected_epochs = max(1, tuned.best_epoch + 1)
    final_average = min(int(options["weight_average_epochs"]), selected_epochs)
    fitted = fit_transformer_interaction_network(
        training,
        width=int(options["width"]),
        num_heads=int(options["num_heads"]),
        num_layers=int(options["num_layers"]),
        epochs=selected_epochs,
        learning_rate=float(options["learning_rate"]),
        weight_decay=float(options["weight_decay"]),
        batch_size=int(options["batch_size"]),
        seed=seed,
        residual=True,
        early_stopping_patience=None,
        weight_average_epochs=final_average,
        num_threads=int(options["num_threads"]),
    )
    predictions = [fitted.predict(batch, speed_cap=speed_cap) for batch in evaluation]
    metadata = {
        "selected_epochs": selected_epochs,
        "tuning_epochs": tuned.epochs_trained,
        "averaged_epochs": fitted.averaged_epochs,
        "parameter_count": fitted.parameter_count,
        "seconds": time.perf_counter() - started,
    }
    return predictions, metadata


def _cached_member_predictions(
    project: Path,
    cache_path: Path,
    training: Sequence[VelocitySamples],
    *,
    validation_index: int,
    evaluation: Sequence[VelocitySamples],
    seed: int,
    options: dict[str, Any],
    speed_cap: float,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    signature = _cache_signature(
        project,
        training,
        validation_index,
        evaluation,
        seed,
        options,
        speed_cap,
    )
    cached = _load_member_cache(
        cache_path,
        signature=signature,
        evaluation=evaluation,
    )
    if cached is not None:
        return cached

    predictions, metadata = _fit_validated_member(
        training,
        validation_index=validation_index,
        evaluation=evaluation,
        seed=seed,
        options=options,
        speed_cap=speed_cap,
    )
    arrays: dict[str, Any] = {
        "cache_schema": np.asarray(_CACHE_SCHEMA),
        "signature": np.asarray(signature),
        "seed": np.asarray(seed),
        "validation_index": np.asarray(validation_index),
        "run_names": np.asarray([batch.run for batch in evaluation]),
        "run_lengths": np.asarray([len(batch.target) for batch in evaluation]),
        **{key: np.asarray(value) for key, value in metadata.items()},
    }
    for index, prediction in enumerate(predictions):
        arrays[f"prediction_{index}"] = np.asarray(prediction, dtype=np.float32)
    _atomic_savez(cache_path, **arrays)
    verified = _load_member_cache(
        cache_path,
        signature=signature,
        evaluation=evaluation,
    )
    if verified is None:
        raise RuntimeError(f"failed to verify Transformer cache {cache_path}")
    return verified


def _fit_fallback(
    project: Path,
    training: Sequence[VelocitySamples],
    evaluation: Sequence[VelocitySamples],
    *,
    seed: int,
    speed_cap: float,
) -> list[np.ndarray]:
    """Fit the identical referee-3 feature-based HGB residual fallback."""

    _, hgb_settings, _ = settings(project)
    model = fit_gradient_boosted_residual(
        list(training),
        anchor(),
        seed=seed,
        **hgb_settings,
    )
    return [model.predict(batch, speed_cap=speed_cap) for batch in evaluation]


def _instantaneous_disagreement(members: np.ndarray) -> np.ndarray:
    ensemble = np.mean(members, axis=0)
    return np.sqrt(np.mean(np.sum((members - ensemble[None, :, :]) ** 2, axis=2), axis=0))


def _causal_score(members: np.ndarray, batch: VelocitySamples, window_seconds: float) -> np.ndarray:
    instantaneous = _instantaneous_disagreement(members)
    return causal_trailing_time_mean(
        instantaneous,
        batch.pedestrian_id,
        batch.frame,
        frames_per_second=25.0,
        window_seconds=window_seconds,
    )


def _metric_rows(
    *,
    regime: str,
    batch: VelocitySamples,
    ensemble: np.ndarray,
    fallback: np.ndarray,
    guarded: np.ndarray,
    score: np.ndarray,
    threshold: float,
    weights: np.ndarray,
) -> list[dict[str, Any]]:
    firing_rate = float(np.mean(score > threshold))
    rows = []
    for model_name, prediction in (
        ("transformer ensemble", ensemble),
        ("feature-based HGB fallback", fallback),
        ("guarded transformer", guarded),
    ):
        error = np.linalg.norm(prediction - batch.target, axis=1)
        rows.append(
            {
                "regime": regime,
                "run": batch.run,
                "model": model_name,
                "n": len(error),
                "rmse_mps": float(np.sqrt(np.mean(error**2))),
                "q95_mps": float(np.quantile(error, 0.95, method="linear")),
                "cvar95_mps": empirical_upper_cvar(error, 0.95),
                "guard_firing_rate": firing_rate,
                "mean_transformer_weight": float(np.mean(weights)),
            }
        )
    return rows


def _write_prediction_bundle(
    path: Path,
    *,
    batch: VelocitySamples,
    role: str,
    members: np.ndarray,
    fallback: np.ndarray,
    score: np.ndarray,
    weights: np.ndarray,
    guarded: np.ndarray,
    threshold: float,
    scale: float,
    seeds: Sequence[int],
) -> None:
    _atomic_savez(
        path,
        output_schema=np.asarray(_OUTPUT_SCHEMA),
        run=np.asarray(batch.run),
        role=np.asarray(role),
        target=np.asarray(batch.target, dtype=np.float32),
        pedestrian_id=np.asarray(batch.pedestrian_id),
        frame=np.asarray(batch.frame),
        member_seeds=np.asarray(seeds),
        members=np.asarray(members, dtype=np.float32),
        ensemble=np.asarray(np.mean(members, axis=0), dtype=np.float32),
        fallback=np.asarray(fallback, dtype=np.float32),
        score=np.asarray(score, dtype=np.float32),
        transformer_weight=np.asarray(weights, dtype=np.float32),
        guarded=np.asarray(guarded, dtype=np.float32),
        threshold=np.asarray(threshold),
        scale=np.asarray(scale),
    )


def run_experiment(args: argparse.Namespace) -> None:
    project = project_root()
    config_path = project / "configs/referee3_lock.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    options = _options(args)
    processed = project / "data/processed"
    cache_dir = processed / "referee3_transformer_cache"
    bundle_dir = processed / "referee3_transformer_predictions"
    cache_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    training, calibration_full, evaluation = build_julich_batches(project, config)
    seeds = _member_seeds(config, 5)
    speed_cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    window_seconds = float(config["guard"]["trailing_window_seconds"])
    quantile = float(config["guard"]["threshold_quantile"])
    base_seed = int(config["replication"]["member_seed_base"])
    manifest_rows: list[dict[str, Any]] = []
    oof_records: list[dict[str, Any]] = []

    # LORO Transformer predictions define the calibration-score distribution.
    for held_index, held_batch in enumerate(calibration_full):
        fold_training = [
            batch for index, batch in enumerate(training) if index != held_index
        ]
        member_predictions = []
        for member_index, seed in enumerate(seeds):
            fold_seed = seed + 100_000 * (held_index + 1)
            validation_index = member_index % len(fold_training)
            cache_path = cache_dir / (
                f"oof_{held_index:02d}_{_safe_name(held_batch.run)}_seed_{fold_seed}.npz"
            )
            prediction, metadata = _cached_member_predictions(
                project,
                cache_path,
                fold_training,
                validation_index=validation_index,
                evaluation=[held_batch],
                seed=fold_seed,
                options=options,
                speed_cap=speed_cap,
            )
            member_predictions.append(prediction[0])
            manifest_rows.append(
                {
                    "fit": "oof",
                    "held_run": held_batch.run,
                    "seed": fold_seed,
                    "validation_run": fold_training[validation_index].run,
                    "cache": str(cache_path.relative_to(project)),
                    "cache_sha256": _sha256(cache_path),
                    **metadata,
                }
            )
            print(f"OOF {held_batch.run}: member {member_index + 1}/5", flush=True)
        members = np.stack(member_predictions)
        fallback = _fit_fallback(
            project,
            fold_training,
            [held_batch],
            seed=base_seed + 700_000 + held_index,
            speed_cap=speed_cap,
        )[0]
        score = _causal_score(members, held_batch, window_seconds)
        oof_records.append(
            {
                "batch": held_batch,
                "members": members,
                "fallback": fallback,
                "score": score,
            }
        )

    calibration_scores = np.concatenate([record["score"] for record in oof_records])
    threshold = finite_sample_upper_quantile(calibration_scores, quantile)
    first_quartile, third_quartile = np.quantile(
        calibration_scores,
        [0.25, 0.75],
        method="linear",
    )
    scale = max(float(third_quartile - first_quartile), 1e-6)
    calibration_path = processed / "referee3_transformer_guard_calibration.json"
    _atomic_write_json(
        calibration_path,
        {
            "schema_version": _OUTPUT_SCHEMA,
            "config_sha256": _sha256(config_path),
            "window_seconds": window_seconds,
            "quantile": quantile,
            "quantile_convention": config["guard"]["quantile_convention"],
            "threshold_mps": threshold,
            "scale_iqr_mps": scale,
            "calibration_forecasts": len(calibration_scores),
            "calibration_firing_rate": float(np.mean(calibration_scores > threshold)),
            "member_seeds": seeds,
            "speed_cap_mps": speed_cap,
            "transformer_options": options,
        },
    )

    for held_index, record in enumerate(oof_records):
        batch = record["batch"]
        members = record["members"]
        fallback = record["fallback"]
        score = record["score"]
        weights = one_sided_gate_weights(
            score,
            threshold=threshold,
            scale=scale,
            kind="exponential",
        )
        ensemble = np.mean(members, axis=0)
        guarded = weights[:, None] * ensemble + (1.0 - weights[:, None]) * fallback
        _write_prediction_bundle(
            bundle_dir / f"oof_{held_index:02d}_{_safe_name(batch.run)}.npz",
            batch=batch,
            role="calibration LORO",
            members=members,
            fallback=fallback,
            score=score,
            weights=weights,
            guarded=guarded,
            threshold=threshold,
            scale=scale,
            seeds=[seed + 100_000 * (held_index + 1) for seed in seeds],
        )

    evaluation_batches = [batch for _, batch in evaluation]
    deployed_by_member: list[list[np.ndarray]] = []
    for member_index, seed in enumerate(seeds):
        validation_index = member_index % len(training)
        cache_path = cache_dir / f"deployed_seed_{seed}.npz"
        predictions, metadata = _cached_member_predictions(
            project,
            cache_path,
            training,
            validation_index=validation_index,
            evaluation=evaluation_batches,
            seed=seed,
            options=options,
            speed_cap=speed_cap,
        )
        deployed_by_member.append(predictions)
        manifest_rows.append(
            {
                "fit": "deployed",
                "held_run": None,
                "seed": seed,
                "validation_run": training[validation_index].run,
                "cache": str(cache_path.relative_to(project)),
                "cache_sha256": _sha256(cache_path),
                **metadata,
            }
        )
        print(f"deployed member {member_index + 1}/5", flush=True)

    fallback_predictions = _fit_fallback(
        project,
        training,
        evaluation_batches,
        seed=base_seed,
        speed_cap=speed_cap,
    )
    metric_rows: list[dict[str, Any]] = []
    for run_index, ((regime, batch), fallback) in enumerate(
        zip(evaluation, fallback_predictions, strict=True)
    ):
        members = np.stack(
            [member_predictions[run_index] for member_predictions in deployed_by_member]
        )
        ensemble = np.mean(members, axis=0)
        score = _causal_score(members, batch, window_seconds)
        weights = one_sided_gate_weights(
            score,
            threshold=threshold,
            scale=scale,
            kind="exponential",
        )
        guarded = weights[:, None] * ensemble + (1.0 - weights[:, None]) * fallback
        metric_rows.extend(
            _metric_rows(
                regime=regime,
                batch=batch,
                ensemble=ensemble,
                fallback=fallback,
                guarded=guarded,
                score=score,
                threshold=threshold,
                weights=weights,
            )
        )
        _write_prediction_bundle(
            bundle_dir / f"test_{_safe_name(regime)}_{_safe_name(batch.run)}.npz",
            batch=batch,
            role=regime,
            members=members,
            fallback=fallback,
            score=score,
            weights=weights,
            guarded=guarded,
            threshold=threshold,
            scale=scale,
            seeds=seeds,
        )
        print(f"evaluated {regime}: {batch.run}", flush=True)

    per_run = pd.DataFrame(metric_rows)
    per_run_path = processed / "referee3_transformer_per_run.csv"
    temporary_csv = per_run_path.with_suffix(".tmp")
    per_run.to_csv(temporary_csv, index=False)
    os.replace(temporary_csv, per_run_path)
    summary = (
        per_run.groupby(["regime", "model"], sort=False)
        .agg(
            runs=("run", "nunique"),
            rmse_mps=("rmse_mps", "mean"),
            q95_mps=("q95_mps", "mean"),
            cvar95_mps=("cvar95_mps", "mean"),
            guard_firing_rate=("guard_firing_rate", "mean"),
            mean_transformer_weight=("mean_transformer_weight", "mean"),
        )
        .reset_index()
    )
    summary_path = processed / "referee3_transformer_regime_summary.csv"
    temporary_summary = summary_path.with_suffix(".tmp")
    summary.to_csv(temporary_summary, index=False)
    os.replace(temporary_summary, summary_path)
    manifest_path = processed / "referee3_transformer_manifest.json"
    _atomic_write_json(
        manifest_path,
        {
            "schema_version": _OUTPUT_SCHEMA,
            "config_sha256": _sha256(config_path),
            "calibration_sha256": _sha256(calibration_path),
            "seeds": seeds,
            "speed_cap_mps": speed_cap,
            "transformer_options": options,
            "fits": manifest_rows,
            "outputs": {
                "per_run": str(per_run_path.relative_to(project)),
                "per_run_sha256": _sha256(per_run_path),
                "regime_summary": str(summary_path.relative_to(project)),
                "regime_summary_sha256": _sha256(summary_path),
            },
        },
    )
    print(summary.to_string(index=False), flush=True)


def _synthetic_batch(name: str, seed: int, n: int = 16) -> VelocitySamples:
    rng = np.random.default_rng(seed)
    features = rng.normal(scale=0.2, size=(n, len(FEATURE_NAMES), 2))
    features[:, 0] += np.array([0.8, 0.0])
    features[:, 1] = np.array([1.0, 0.0])
    features[:, 5] = 0.0
    neighbours = np.zeros((n, 4, 5))
    neighbours[:, 0, :4] = rng.normal(scale=0.2, size=(n, 4))
    neighbours[:, 0, 0] += 0.8
    neighbours[:, 0, 4] = 1.0
    target = features[:, 0] + np.column_stack(
        (0.05 * np.tanh(neighbours[:, 0, 0]), -0.03 * np.tanh(neighbours[:, 0, 1]))
    )
    return VelocitySamples(
        name,
        features,
        target,
        np.arange(n) % 4,
        np.repeat(np.arange(n // 4), 4) * 10,
        raw_neighbours=neighbours,
        origin_time_s=np.repeat(np.arange(n // 4), 4) * 0.4,
    )


def dry_run() -> None:
    training = [_synthetic_batch("train_a", 1), _synthetic_batch("train_b", 2)]
    evaluation = _synthetic_batch("evaluation", 3)
    options = {
        "width": 8,
        "num_heads": 2,
        "num_layers": 1,
        "max_epochs": 2,
        "patience": 1,
        "weight_average_epochs": 1,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 1e-3,
        "num_threads": 1,
        "residual": True,
        "wall_features": False,
    }
    member_predictions = []
    for member_index, seed in enumerate((101, 202)):
        prediction, _ = _fit_validated_member(
            training,
            validation_index=member_index,
            evaluation=[evaluation],
            seed=seed,
            options=options,
            speed_cap=3.0,
        )
        member_predictions.append(prediction[0])
    members = np.stack(member_predictions)
    score = _causal_score(members, evaluation, 1.6)
    threshold = finite_sample_upper_quantile(score, 0.9)
    scale = max(float(np.subtract(*np.quantile(score, [0.75, 0.25]))), 1e-6)
    weight = one_sided_gate_weights(score, threshold=threshold, scale=scale)
    if members.shape != (2, len(evaluation.target), 2):
        raise RuntimeError("synthetic member stack has an unexpected shape")
    if not np.all(np.isfinite(weight)):
        raise RuntimeError("synthetic guard produced non-finite weights")
    print(
        json.dumps(
            {
                "status": "ok",
                "members": members.shape[0],
                "forecasts": members.shape[1],
                "threshold": threshold,
                "scale": scale,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--weight-average-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
