# Author: Amir Ghorbani
"""Cohort-conditioned 3.2 s autoregressive rollout on held-out crossing runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pedgeom.benchmarks import (
    DisagreementRoutedRegressor,
    GELPedRegressor,
    calibrate_disagreement_router,
)
from pedgeom.calibration import FEATURE_NAMES, VelocitySamples, _interaction_features
from pedgeom.datasets import (
    add_motion_features,
    add_prediction_goal_directions,
    load_julich_crossing_trajectory,
)

from major_revision_analysis import ModelSpec, load_inputs, project_root
from reviewer_revision_analysis import build_datasets
from shift_routed_upgrade import fit_routed_models
from anticipatory_residual_upgrade import fit_hgb


STEP_FRAMES = 10
STEP_SECONDS = 0.4
ROLLOUT_STEPS = 8
CONFLICT_DISTANCE_M = 0.5


def candidate_scenes(source: Path, maximum_scenes: int = 12) -> list[dict]:
    data = add_prediction_goal_directions(
        add_motion_features(load_julich_crossing_trajectory(source)),
        method="entry",
        history_frames=STEP_FRAMES,
    )
    indexed = data.set_index(["frame", "pedestrian_id"]).sort_index()
    frames = np.sort(data["frame"].unique())
    scenes = []
    last_start = -10**9
    for frame in frames:
        if frame - last_start < 4 * STEP_FRAMES:
            continue
        current = data[(data["frame"] == frame) & data["goal_valid"]]
        ids = set(current["pedestrian_id"].astype(int))
        for offset in (-STEP_FRAMES, *range(STEP_FRAMES, (ROLLOUT_STEPS + 1) * STEP_FRAMES, STEP_FRAMES)):
            present = set(data.loc[data["frame"] == frame + offset, "pedestrian_id"].astype(int))
            ids &= present
        if len(ids) < 4:
            continue
        ordered_ids = np.array(sorted(ids), dtype=int)

        def positions(at_frame: int) -> np.ndarray:
            return indexed.loc[(at_frame, ordered_ids), ["x_m", "y_m"]].to_numpy(float)

        initial = positions(frame)
        previous = positions(frame - STEP_FRAMES)
        goals = indexed.loc[(frame, ordered_ids), ["goal_x", "goal_y"]].to_numpy(float)
        truth = np.stack(
            [positions(frame + step * STEP_FRAMES) for step in range(ROLLOUT_STEPS + 1)]
        )
        scenes.append(
            {
                "run": source.stem,
                "start_frame": int(frame),
                "ids": ordered_ids,
                "position": initial,
                "velocity": (initial - previous) / STEP_SECONDS,
                "goal": goals,
                "truth": truth,
            }
        )
        last_start = int(frame)
        if len(scenes) >= maximum_scenes:
            break
    return scenes


def model_samples(
    run: str,
    frame: int,
    ids: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
) -> VelocitySamples:
    isotropic, forward, closing, wall, neighbour_count = _interaction_features(
        position,
        velocity,
        goal,
        interaction_range=0.8,
        radius=0.25,
        wall_range=0.25,
        wall_y_bounds=None,
        neighbour_cutoff=3.0,
    )
    features = np.zeros((len(ids), len(FEATURE_NAMES), 2))
    features[:, 0] = velocity
    features[:, 1] = goal
    features[:, 2] = isotropic
    features[:, 3] = forward
    features[:, 4] = closing
    features[:, 5] = wall
    # Same deterministic current-state neighbour representation used in model
    # fitting.  Recomputing it after each predicted step lets the prospective
    # conflict state evolve without accessing future observations.
    maximum_neighbours = 8
    delta = position[None, :, :] - position[:, None, :]
    relative_velocity = velocity[None, :, :] - velocity[:, None, :]
    distance = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distance, np.inf)
    raw_neighbours = np.zeros((len(ids), maximum_neighbours, 5), dtype=float)
    for focal in range(len(ids)):
        ordered = np.argsort(distance[focal])
        ordered = ordered[distance[focal, ordered] < 3.0][:maximum_neighbours]
        raw_neighbours[focal, : len(ordered), :2] = delta[focal, ordered]
        raw_neighbours[focal, : len(ordered), 2:4] = relative_velocity[
            focal, ordered
        ]
        raw_neighbours[focal, : len(ordered), 4] = 1.0
    return VelocitySamples(
        run,
        features,
        np.zeros_like(position),
        ids,
        np.full(len(ids), frame),
        position,
        neighbour_count.astype(float),
        np.full(len(ids), len(ids), dtype=float),
        "entry",
        raw_neighbours,
    )


def rollout(scene: dict, model: object, speed_cap: float) -> np.ndarray:
    position = scene["position"].copy()
    velocity = scene["velocity"].copy()
    trajectory = [position.copy()]
    for step in range(ROLLOUT_STEPS):
        samples = model_samples(
            scene["run"],
            scene["start_frame"] + step * STEP_FRAMES,
            scene["ids"],
            position,
            velocity,
            scene["goal"],
        )
        velocity = model.predict(samples, speed_cap=speed_cap)
        position = position + STEP_SECONDS * velocity
        trajectory.append(position.copy())
    return np.stack(trajectory)


def conflict_vector(trajectory: np.ndarray) -> np.ndarray:
    count = trajectory.shape[1]
    first, second = np.triu_indices(count, k=1)
    distance = np.linalg.norm(
        trajectory[:, first, :] - trajectory[:, second, :], axis=2
    )
    return np.min(distance, axis=0) < CONFLICT_DISTANCE_M


def main() -> None:
    project = project_root()
    processed = project / "data" / "processed"
    config, splits = load_inputs(project)
    selected = json.loads(
        (processed / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    training, evaluation = build_datasets(project, config, splits)
    hgb_selection = json.loads(
        (processed / "shift_routed_hgb_selection.json").read_text(encoding="utf-8")
    )
    fitted = fit_routed_models(
        list(training.values()), selected, config, hgb_selection
    )
    prospective_selection = json.loads(
        (processed / "prospective_selection.json").read_text(encoding="utf-8")
    )
    prospective_parameters = {
        key: prospective_selection[key]
        for key in ("horizon_s", "time_scale_s", "clearance_scale_m")
    }
    prospective_settings = {
        key: value
        for key, value in hgb_selection["anchored_residual"].items()
        if key != "candidate"
    }
    direct_settings = {
        key: value
        for key, value in hgb_selection["direct"].items()
        if key != "candidate"
    }
    training_batches = list(training.values())
    prospective_direct = fit_hgb(
        training_batches,
        prospective_parameters,
        direct_settings,
        config["seed"],
        residual=False,
    )
    prospective_residual = fit_hgb(
        training_batches,
        prospective_parameters,
        prospective_settings,
        config["seed"],
        residual=True,
    )
    direct_anticipatory_control = calibrate_disagreement_router(
        training_batches, prospective_direct, fitted["direct_mlp"].model
    )
    gel_ped_anticipatory = calibrate_disagreement_router(
        training_batches,
        prospective_residual,
        fitted["kinematic_residual_mlp"].model,
    )
    models = {
        "constant_velocity": fitted["constant_velocity"],
        "direct_mlp": fitted["direct_mlp"],
        "direct_anticipatory_control": ModelSpec(
            direct_anticipatory_control, selected["speed_cap_mps"], -1
        ),
        "tensor_prior": fitted["tensor_prior"],
        "gel_ped_v1": fitted["gel_ped"],
        "gel_ped_anticipatory": ModelSpec(
            gel_ped_anticipatory, selected["speed_cap_mps"], -1
        ),
    }
    crossing_batches = {
        batch.run: batch for batch in evaluation["crossing topology"]
    }
    crossing = project / "data" / "raw" / "2013crossing90" / "trajectories"
    rows = []
    conflict_rows = []
    manifest = []
    for source in sorted(crossing.glob("crossing_90_[de]_*.txt")):
        run_models = dict(models)
        for model_name, spec in list(run_models.items()):
            if not isinstance(spec.model, DisagreementRoutedRegressor):
                continue
            router = models[model_name].model
            weight = router.in_support_weight(crossing_batches[source.stem])
            run_models[model_name] = ModelSpec(
                GELPedRegressor(router.in_support, router.transfer, weight),
                models[model_name].speed_cap,
                models[model_name].parameters,
            )
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
            for model_name, spec in run_models.items():
                prediction = rollout(scene, spec.model, spec.speed_cap)
                distance_error = np.linalg.norm(prediction[1:] - truth[1:], axis=2)
                for step in range(1, ROLLOUT_STEPS + 1):
                    rows.append(
                        {
                            "run": source.stem,
                            "start_frame": scene["start_frame"],
                            "model": model_name,
                            "horizon_s": step * STEP_SECONDS,
                            "pedestrians": len(scene["ids"]),
                            "mean_displacement_error_m": float(
                                np.mean(distance_error[step - 1])
                            ),
                        }
                    )
                predicted_conflict = conflict_vector(prediction)
                conflict_rows.append(
                    {
                        "run": source.stem,
                        "start_frame": scene["start_frame"],
                        "model": model_name,
                        "pairs": len(true_conflict),
                        "true_positive": int(np.sum(true_conflict & predicted_conflict)),
                        "false_positive": int(np.sum(~true_conflict & predicted_conflict)),
                        "false_negative": int(np.sum(true_conflict & ~predicted_conflict)),
                        "true_negative": int(np.sum(~true_conflict & ~predicted_conflict)),
                    }
                )
        print(f"rollout complete: {source.stem} ({len(scenes)} scenes)", flush=True)

    horizon_metrics = pd.DataFrame(rows)
    horizon_metrics.to_csv(processed / "autoregressive_rollout_horizon_metrics.csv", index=False)
    run_horizon = (
        horizon_metrics.groupby(["run", "model", "horizon_s"], sort=False)
        .apply(
            lambda frame: np.average(
                frame["mean_displacement_error_m"], weights=frame["pedestrians"]
            ),
            include_groups=False,
        )
        .rename("mean_displacement_error_m")
        .reset_index()
    )
    run_horizon.to_csv(processed / "autoregressive_rollout_run_horizon.csv", index=False)
    summary = (
        run_horizon.groupby(["model", "horizon_s"], sort=False)[
            "mean_displacement_error_m"
        ]
        .agg(mean_m="mean", sd_m="std", runs="count")
        .reset_index()
    )
    summary.to_csv(processed / "autoregressive_rollout_summary.csv", index=False)

    conflicts = pd.DataFrame(conflict_rows)
    conflicts.to_csv(processed / "autoregressive_conflict_metrics.csv", index=False)
    conflict_run = conflicts.groupby(["run", "model"], sort=False)[
        ["true_positive", "false_positive", "false_negative", "true_negative"]
    ].sum()
    conflict_run["precision"] = conflict_run["true_positive"] / np.maximum(
        conflict_run["true_positive"] + conflict_run["false_positive"], 1
    )
    conflict_run["recall"] = conflict_run["true_positive"] / np.maximum(
        conflict_run["true_positive"] + conflict_run["false_negative"], 1
    )
    conflict_run["f1"] = (
        2.0
        * conflict_run["precision"]
        * conflict_run["recall"]
        / np.maximum(conflict_run["precision"] + conflict_run["recall"], 1e-12)
    )
    conflict_run.reset_index().to_csv(
        processed / "autoregressive_conflict_run_summary.csv", index=False
    )
    (
        conflict_run.groupby("model")[["precision", "recall", "f1"]]
        .agg(["mean", "std"])
        .to_csv(processed / "autoregressive_conflict_summary.csv")
    )
    (processed / "autoregressive_rollout_manifest.json").write_text(
        json.dumps(
            {
                "step_seconds": STEP_SECONDS,
                "steps": ROLLOUT_STEPS,
                "horizon_seconds": STEP_SECONDS * ROLLOUT_STEPS,
                "scene_spacing_seconds": 4 * STEP_SECONDS,
                "conflict_distance_m": CONFLICT_DISTANCE_M,
                "cohort_conditioned": True,
                "runs": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("autoregressive rollout analysis complete", flush=True)


if __name__ == "__main__":
    main()
