# Author: Amir Ghorbani
"""Operational, sensitivity, uncertainty, and rollout audits for GEL-Ped."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pedgeom.calibration import VelocitySamples, velocity_metrics

from autoregressive_rollout import (
    ROLLOUT_STEPS,
    STEP_FRAMES,
    STEP_SECONDS,
    candidate_scenes,
    conflict_vector,
    model_samples,
)
from causal_anchor_revision import build_julich, conformal_quantile, load_inputs, project_root, settings
from referee2_revision import (
    _base_predictions,
    _fit_mahalanobis,
    _fit_models,
    _gate_weight,
    _rolling_by_pedestrian,
    calibrate,
)


PRIMARY = "GEL-Ped instability guard"


def cap_speed(prediction: np.ndarray, cap: float = 2.5) -> np.ndarray:
    speed = np.linalg.norm(prediction, axis=1)
    scale = np.minimum(1.0, cap / np.maximum(speed, 1e-12))
    return prediction * scale[:, None]


def primary_prediction(
    base: dict[str, np.ndarray],
    samples: VelocitySamples,
    threshold: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = _rolling_by_pedestrian(base["epistemic"], samples)
    weight = _gate_weight(score, threshold, scale, 1.0)
    prediction = weight[:, None] * base["ensemble"] + (
        1.0 - weight[:, None]
    ) * base["coarse"]
    return prediction, weight, score


def unsafe(
    batch: VelocitySamples,
    focal_velocity: np.ndarray,
    *,
    margin_m: float = 0.8,
    horizon_s: float = 1.0,
) -> np.ndarray:
    raw = batch.raw_neighbours
    position = raw[:, :, :2]
    present = raw[:, :, 4] > 0.5
    neighbour_velocity = batch.features[:, None, 0, :] + raw[:, :, 2:4]
    relative_velocity = neighbour_velocity - focal_velocity[:, None, :]
    speed_squared = np.sum(relative_velocity**2, axis=2)
    time = np.divide(
        -np.sum(position * relative_velocity, axis=2),
        speed_squared,
        out=np.zeros_like(speed_squared),
        where=speed_squared > 1e-9,
    )
    time = np.clip(time, 0.0, horizon_s)
    distance = np.linalg.norm(position + time[:, :, None] * relative_velocity, axis=2)
    distance[~present] = np.inf
    return np.min(distance, axis=1) < margin_m


def mondrian_calibration(
    oof: pd.DataFrame, threshold: float, scale: float
) -> tuple[float, dict[int, float]]:
    graph = oof[["graph_x", "graph_y"]].to_numpy()
    coarse = oof[["coarse_x", "coarse_y"]].to_numpy()
    target = oof[["target_x", "target_y"]].to_numpy()
    score = oof["local_epistemic"].to_numpy()
    weight = _gate_weight(score, threshold, scale, 1.0)
    prediction = weight[:, None] * graph + (1.0 - weight[:, None]) * coarse
    error = np.linalg.norm(prediction - target, axis=1)
    normalized = (score - threshold) / max(scale, 1e-9)
    code = np.digitize(normalized, [0.0, 1.0, 2.0])
    global_radius = conformal_quantile(error, 0.90)
    radii = {
        value: conformal_quantile(error[code == value], 0.90)
        for value in range(4)
        if np.any(code == value)
    }
    return global_radius, radii


def one_step_audits(
    crossing: list[VelocitySamples],
    models: object,
    oof: pd.DataFrame,
    threshold: float,
    scale: float,
    processed: Path,
) -> None:
    quantiles = {
        "q80": conformal_quantile(oof["local_epistemic"].to_numpy(), 0.80),
        "q90": conformal_quantile(oof["local_epistemic"].to_numpy(), 0.90),
        "q95": conformal_quantile(oof["local_epistemic"].to_numpy(), 0.95),
    }
    sensitivity_rows: list[dict] = []
    collision_rows: list[dict] = []
    oracle_rows: list[dict] = []
    coverage_rows: list[dict] = []
    global_radius, conditional_radii = mondrian_calibration(oof, threshold, scale)
    for batch in crossing:
        base = _base_predictions(models, batch)
        primary, weight, score = primary_prediction(base, batch, threshold, scale)
        for label, candidate_threshold in quantiles.items():
            for scale_multiplier in (0.5, 1.0, 2.0):
                candidate, candidate_weight, _ = primary_prediction(
                    base,
                    batch,
                    candidate_threshold,
                    scale * scale_multiplier,
                )
                sensitivity_rows.append(
                    {
                        "run": batch.run,
                        "threshold": label,
                        "scale_multiplier": scale_multiplier,
                        "mean_graph_weight": float(np.mean(candidate_weight)),
                        **velocity_metrics(batch.target, candidate),
                    }
                )

        predictions = {
            "constant velocity": base["cv"],
            "five-member graph ensemble": base["ensemble"],
            "coordinate-median graph ensemble": base["median"],
            "anchored coarse HGB": base["coarse"],
            PRIMARY: primary,
        }
        truth = unsafe(batch, batch.target)
        for name, prediction in predictions.items():
            flagged = unsafe(batch, prediction)
            collision_rows.append(
                {
                    "run": batch.run,
                    "model": name,
                    "true_positive": int(np.sum(truth & flagged)),
                    "false_positive": int(np.sum(~truth & flagged)),
                    "false_negative": int(np.sum(truth & ~flagged)),
                    "true_negative": int(np.sum(~truth & ~flagged)),
                }
            )

        frames = np.unique(batch.frame)
        grid = np.linspace(0.0, 1.0, 21)
        graph = base["ensemble"]
        coarse = base["coarse"]
        buffer_sse = 0.0
        buffer_samples = 0
        chosen_weights: list[float] = []
        for start in range(0, len(frames), 5):
            selected = np.isin(batch.frame, frames[start : start + 5])
            losses = [
                np.sum(
                    (
                        value * graph[selected]
                        + (1.0 - value) * coarse[selected]
                        - batch.target[selected]
                    )
                    ** 2
                )
                for value in grid
            ]
            best = int(np.argmin(losses))
            chosen_weights.append(float(grid[best]))
            buffer_sse += float(losses[best])
            buffer_samples += int(np.sum(selected))
        oracle_rows.append(
            {
                "run": batch.run,
                "model": "label-using five-frame buffer oracle",
                "vector_rmse_mps": float(np.sqrt(buffer_sse / buffer_samples)),
                "mean_graph_weight": float(np.mean(chosen_weights)),
                "buffers": len(chosen_weights),
            }
        )
        oracle_rows.append(
            {
                "run": batch.run,
                "model": PRIMARY,
                "vector_rmse_mps": velocity_metrics(batch.target, primary)[
                    "vector_rmse_mps"
                ],
                "mean_graph_weight": float(np.mean(weight)),
                "buffers": len(chosen_weights),
            }
        )

        normalized = (score - threshold) / max(scale, 1e-9)
        code = np.digitize(normalized, [0.0, 1.0, 2.0])
        radial_error = np.linalg.norm(primary - batch.target, axis=1)
        conditional = np.array([conditional_radii[int(value)] for value in code])
        for value, label in enumerate(("inside", "0--1 IQR", "1--2 IQR", ">2 IQR")):
            selected = code == value
            if not np.any(selected):
                continue
            coverage_rows.append(
                {
                    "run": batch.run,
                    "stratum": label,
                    "samples": int(np.sum(selected)),
                    "global_radius_mps": global_radius,
                    "conditional_radius_mps": conditional_radii[value],
                    "global_coverage": float(
                        np.mean(radial_error[selected] <= global_radius)
                    ),
                    "conditional_coverage": float(
                        np.mean(radial_error[selected] <= conditional[selected])
                    ),
                }
            )
    pd.DataFrame(sensitivity_rows).to_csv(
        processed / "referee2_threshold_sensitivity.csv", index=False
    )
    pd.DataFrame(collision_rows).to_csv(
        processed / "referee2_collision_counts.csv", index=False
    )
    pd.DataFrame(oracle_rows).to_csv(
        processed / "referee2_buffer_oracle.csv", index=False
    )
    pd.DataFrame(coverage_rows).to_csv(
        processed / "referee2_mondrian_coverage.csv", index=False
    )


def rollout_scene(
    scene: dict,
    models: object,
    model_name: str,
    threshold: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    position = scene["position"].copy()
    velocity = scene["velocity"].copy()
    trajectory = [position.copy()]
    score_history: list[np.ndarray] = []
    weights: list[float] = []
    for step in range(ROLLOUT_STEPS):
        samples = model_samples(
            scene["run"],
            scene["start_frame"] + step * STEP_FRAMES,
            scene["ids"],
            position,
            velocity,
            scene["goal"],
        )
        base = _base_predictions(models, samples)
        if model_name == "constant velocity":
            velocity = base["cv"]
        elif model_name == "five-member graph ensemble":
            velocity = base["ensemble"]
        elif model_name == "coordinate-median graph ensemble":
            velocity = base["median"]
        elif model_name == "anchored coarse HGB":
            velocity = base["coarse"]
        elif model_name == PRIMARY:
            score_history.append(base["epistemic"])
            rolling = np.mean(score_history[-5:], axis=0)
            weight = _gate_weight(rolling, threshold, scale, 1.0)
            weights.append(float(np.mean(weight)))
            velocity = weight[:, None] * base["ensemble"] + (
                1.0 - weight[:, None]
            ) * base["coarse"]
        else:
            raise ValueError(model_name)
        velocity = cap_speed(velocity)
        position = position + STEP_SECONDS * velocity
        trajectory.append(position.copy())
    return np.stack(trajectory), np.asarray(weights)


def rollout_audit(
    project: Path,
    models: object,
    threshold: float,
    scale: float,
    processed: Path,
) -> None:
    names = (
        "constant velocity",
        "five-member graph ensemble",
        "coordinate-median graph ensemble",
        "anchored coarse HGB",
        PRIMARY,
    )
    crossing = project / "data/raw/2013crossing90/trajectories"
    rows: list[dict] = []
    conflict_rows: list[dict] = []
    manifest: list[dict] = []
    for source in sorted(crossing.glob("crossing_90_[de]_*.txt")):
        scenes = candidate_scenes(source)
        manifest.append(
            {
                "run": source.stem,
                "scenes": len(scenes),
                "pedestrian_forecasts": int(sum(len(scene["ids"]) for scene in scenes)),
            }
        )
        for scene in scenes:
            truth = scene["truth"]
            true_conflict = conflict_vector(truth)
            for name in names:
                prediction, weights = rollout_scene(
                    scene, models, name, threshold, scale
                )
                error = np.linalg.norm(prediction[1:] - truth[1:], axis=2)
                for step in range(1, ROLLOUT_STEPS + 1):
                    rows.append(
                        {
                            "run": source.stem,
                            "start_frame": scene["start_frame"],
                            "model": name,
                            "horizon_s": step * STEP_SECONDS,
                            "pedestrians": len(scene["ids"]),
                            "mean_displacement_error_m": float(
                                np.mean(error[step - 1])
                            ),
                            "mean_graph_weight": float(np.mean(weights[:step]))
                            if len(weights)
                            else np.nan,
                        }
                    )
                predicted_conflict = conflict_vector(prediction)
                conflict_rows.append(
                    {
                        "run": source.stem,
                        "start_frame": scene["start_frame"],
                        "model": name,
                        "pairs": len(true_conflict),
                        "true_positive": int(np.sum(true_conflict & predicted_conflict)),
                        "false_positive": int(np.sum(~true_conflict & predicted_conflict)),
                        "false_negative": int(np.sum(true_conflict & ~predicted_conflict)),
                        "true_negative": int(np.sum(~true_conflict & ~predicted_conflict)),
                    }
                )
        print(f"rollout complete: {source.stem}", flush=True)
    pd.DataFrame(rows).to_csv(processed / "referee2_rollout_metrics.csv", index=False)
    pd.DataFrame(conflict_rows).to_csv(
        processed / "referee2_rollout_conflicts.csv", index=False
    )
    (processed / "referee2_rollout_manifest.json").write_text(
        json.dumps(
            {
                "step_seconds": STEP_SECONDS,
                "steps": ROLLOUT_STEPS,
                "horizon_seconds": STEP_SECONDS * ROLLOUT_STEPS,
                "scene_spacing_seconds": 4 * STEP_SECONDS,
                "cohort_conditioned": True,
                "runs": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    project = project_root()
    processed = project / "data/processed"
    config, splits = load_inputs(project)
    _, hgb, prospective = settings(project)
    training_map, datasets = build_julich(project, config, splits, history_frames=20)
    training = list(training_map.values())
    seeds = [config["seed"] + 997 * index for index in range(5)]
    models = _fit_models(training, seeds, 25, hgb, prospective)
    _fit_mahalanobis(training)  # verifies the OOD comparator remains estimable
    oof = pd.read_csv(processed / "referee2_oof_calibration.csv")
    calibrations, _ = calibrate(oof)
    calibration = calibrations["coarse"]
    one_step_audits(
        datasets["crossing topology"],
        models,
        oof,
        calibration.epistemic_threshold,
        calibration.epistemic_scale,
        processed,
    )
    rollout_audit(
        project,
        models,
        calibration.epistemic_threshold,
        calibration.epistemic_scale,
        processed,
    )


if __name__ == "__main__":
    main()
