# Author: Amir Ghorbani
"""Paired reactive-planner replay using saved referee-3 prediction bundles.

The experiment compares three policies with identical recorded crowds and
robot routes: direct graph forecasts with a fixed margin, guarded forecasts
with the same fixed margin, and guarded forecasts with the locked g-adaptive
margin.  Pedestrians follow recorded motion; no reciprocal response or
closed-loop human behaviour is claimed.

Only post-prediction aggregation and planner search can be timed from saved
bundles.  The emitted latency table labels that scope explicitly.  Actual
network/HGB inference should be measured with
``pedgeom.reactive_planner.benchmark_model_stack`` while trained model objects
are resident in the full generation pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd

from pedgeom.reactive_planner import (
    PlannerParameters,
    ReplayFrame,
    benchmark_callable,
    choose_reactive_velocity,
    derive_replay_route,
    replay_episode,
)
from pedgeom.runtime_guard import one_sided_gate_weights
from referee3_analyse import _manifest_bundles, selected_predictions, signals
from referee3_generate_predictions import _atomic_write_json, _sha256, project_root


def _atomic_csv(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _predictions(bundle: Any, calibration: dict[str, Any], config: dict[str, Any]):
    indices = tuple(int(index) for index in calibration["indices"])
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    members, direct, fallback = selected_predictions(bundle, indices, cap)
    current = signals(
        bundle,
        members,
        direct,
        fallback,
        float(config["guard"]["trailing_window_seconds"]),
    )
    graph_weight = one_sided_gate_weights(
        current["spread"],
        threshold=float(calibration["tau"]),
        scale=float(calibration["scale"]),
        kind=config["guard"]["primary_shape"],
    )
    guarded = graph_weight[:, None] * direct + (1.0 - graph_weight[:, None]) * fallback
    return members, direct, fallback, guarded, graph_weight


def _frames(
    bundle: Any,
    forecast: np.ndarray,
    graph_weight: np.ndarray,
) -> list[ReplayFrame]:
    frames = []
    for frame in np.unique(bundle.frame):
        selected = bundle.frame == frame
        frames.append(
            ReplayFrame(
                frame=float(frame),
                pedestrian_positions=np.asarray(bundle.position[selected], dtype=float),
                forecast_velocities=np.asarray(forecast[selected], dtype=float),
                realised_velocities=np.asarray(bundle.target[selected], dtype=float),
                graph_weights=np.asarray(graph_weight[selected], dtype=float),
            )
        )
    return frames


def _result_row(role: str, run: str, policy: str, result: Any) -> dict[str, Any]:
    return {
        "role": role,
        "run": run,
        "policy": policy,
        "completed": result.completed,
        "failure": result.failure,
        "steps": result.steps,
        "elapsed_seconds": result.elapsed_seconds,
        "travel_time_seconds": result.travel_time_seconds,
        "route_length_m": result.route_length_m,
        "travelled_distance_m": result.travelled_distance_m,
        "progress_fraction": result.progress_fraction,
        "personal_space_intrusions": result.personal_space_intrusions,
        "intrusion_rate": result.intrusion_rate,
        "minimum_separation_m": result.minimum_separation_m,
        "stop_count": result.stop_count,
        "stop_frequency": result.stop_frequency,
        "infeasible_count": result.infeasible_count,
    }


def _trace_rows(role: str, run: str, policy: str, result: Any) -> list[dict[str, Any]]:
    rows = []
    for index, step in enumerate(result.trace):
        rows.append(
            {
                "role": role,
                "run": run,
                "policy": policy,
                "step": index,
                **step.__dict__,
            }
        )
    return rows


def _latency_rows(
    bundle: Any,
    members: np.ndarray,
    fallback: np.ndarray,
    guarded: np.ndarray,
    graph_weight: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    parameters: PlannerParameters,
    calibration: dict[str, Any],
    *,
    warmup: int,
    repeats: int,
) -> list[dict[str, Any]]:
    unique_frames = np.unique(bundle.frame)
    counts = np.array([np.sum(bundle.frame == frame) for frame in unique_frames])
    chosen_frame = unique_frames[int(np.argsort(counts)[len(counts) // 2])]
    selected = bundle.frame == chosen_frame
    member_frame = members[:, selected]
    fallback_frame = fallback[selected]
    position_frame = bundle.position[selected]
    guarded_frame = guarded[selected]
    weight_frame = graph_weight[selected]

    def aggregate_direct() -> np.ndarray:
        return np.mean(member_frame, axis=0)

    def guard_and_blend() -> np.ndarray:
        direct = np.mean(member_frame, axis=0)
        spread = np.sqrt(
            np.mean(np.sum((member_frame - direct[None, :, :]) ** 2, axis=2), axis=0)
        )
        weight = one_sided_gate_weights(
            spread,
            threshold=float(calibration["tau"]),
            scale=float(calibration["scale"]),
        )
        return weight[:, None] * direct + (1.0 - weight[:, None]) * fallback_frame

    direct_frame = np.mean(member_frame, axis=0)

    def direct_plan():
        return choose_reactive_velocity(
            start,
            goal,
            position_frame,
            direct_frame,
            parameters.fixed_clearance_m,
            parameters,
        )

    def guarded_plan():
        clearance = parameters.fixed_clearance_m + parameters.adaptive_extra_clearance_m * (
            1.0 - weight_frame
        )
        return choose_reactive_velocity(
            start,
            goal,
            position_frame,
            guarded_frame,
            clearance,
            parameters,
        )

    rows = []
    for stage, function in (
        ("five-member reduction", aggregate_direct),
        ("instantaneous guard and blend", guard_and_blend),
        ("reactive search with direct forecast", direct_plan),
        ("reactive search with guarded adaptive forecast", guarded_plan),
    ):
        rows.append(
            {
                "role": bundle.role,
                "run": bundle.run,
                "stage": stage,
                "scope": "post-prediction CPU timing; excludes neural and HGB forward passes",
                "pedestrians": int(np.sum(selected)),
                **benchmark_callable(function, warmup=warmup, repeats=repeats),
            }
        )
    return rows


def evaluate(args: argparse.Namespace) -> None:
    project = project_root()
    config_path = project / "configs/referee3_lock.json"
    calibration_path = project / "data/processed/referee3_primary_calibration.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not calibration_path.exists():
        raise RuntimeError("run referee3_analyse.py before the planner audit")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    parameters = PlannerParameters.from_mapping(config["reactive_planner"])
    _, bundles = _manifest_bundles(project)
    if not bundles:
        raise RuntimeError("no referee-3 test prediction bundles were found")
    bundle_paths = [bundle.path for bundle in bundles]

    metric_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        members, direct, fallback, guarded, graph_weight = _predictions(
            bundle, calibration, config
        )
        first_frame = np.min(bundle.frame)
        first_positions = bundle.position[bundle.frame == first_frame]
        start, goal = derive_replay_route(
            first_positions,
            padding_m=parameters.fixed_clearance_m,
        )
        route_rows.append(
            {
                "role": bundle.role,
                "run": bundle.run,
                "first_frame": int(first_frame),
                "start_x": float(start[0]),
                "start_y": float(start[1]),
                "goal_x": float(goal[0]),
                "goal_y": float(goal[1]),
                "route_definition": "minor spatial axis of first observed frame",
            }
        )
        policies = (
            ("direct forecast, fixed margin", direct, np.ones(len(direct)), False),
            ("guarded forecast, fixed margin", guarded, graph_weight, False),
            ("guarded forecast, g-adaptive margin", guarded, graph_weight, True),
        )
        for policy, forecast, weights, adaptive in policies:
            result = replay_episode(
                _frames(bundle, forecast, weights),
                start,
                goal,
                parameters,
                adaptive_margin=adaptive,
            )
            metric_rows.append(_result_row(bundle.role, bundle.run, policy, result))
            trace_rows.extend(_trace_rows(bundle.role, bundle.run, policy, result))
        latency_rows.extend(
            _latency_rows(
                bundle,
                members,
                fallback,
                guarded,
                graph_weight,
                start,
                goal,
                parameters,
                calibration,
                warmup=args.latency_warmup,
                repeats=args.latency_repeats,
            )
        )
        print(f"planner replay complete: {bundle.role} / {bundle.run}", flush=True)

    output = project / "data/processed"
    metrics_path = output / "referee3_planner_metrics.csv"
    trace_path = output / "referee3_planner_trace.csv.gz"
    latency_path = output / "referee3_planner_latency.csv"
    route_path = output / "referee3_planner_routes.csv"
    _atomic_csv(pd.DataFrame(metric_rows), metrics_path)
    _atomic_csv(pd.DataFrame(trace_rows), trace_path, compression="gzip")
    _atomic_csv(pd.DataFrame(latency_rows), latency_path)
    _atomic_csv(pd.DataFrame(route_rows), route_path)
    manifest_path = output / "referee3_planner_manifest.json"
    _atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "scope": config["reactive_planner"]["scope"],
            "episode_policy": config["reactive_planner"]["episode_policy"],
            "route_definition": "minor spatial axis of first observed frame",
            "latency_scope": (
                "post-prediction CPU timing only; use benchmark_model_stack "
                "with resident trained models for full inference latency"
            ),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "config_sha256": _sha256(config_path),
            "calibration_sha256": _sha256(calibration_path),
            "bundles": [
                {"path": str(path.relative_to(project)), "sha256": _sha256(path)}
                for path in bundle_paths
            ],
            "outputs": {
                "metrics": _sha256(metrics_path),
                "trace": _sha256(trace_path),
                "latency": _sha256(latency_path),
                "routes": _sha256(route_path),
            },
        },
    )


def dry_run() -> None:
    project = project_root()
    config = json.loads((project / "configs/referee3_lock.json").read_text())
    parameters = PlannerParameters.from_mapping(config["reactive_planner"])
    positions = np.array([[-1.0, -0.2], [0.0, 0.0], [1.0, 0.2]])
    start, goal = derive_replay_route(positions, padding_m=parameters.fixed_clearance_m)
    frames = [
        ReplayFrame(
            frame=float(index),
            pedestrian_positions=positions,
            forecast_velocities=np.zeros_like(positions),
            realised_velocities=np.zeros_like(positions),
            graph_weights=np.array([1.0, 0.7, 0.4]),
        )
        for index in range(3)
    ]
    fixed = replay_episode(frames, start, goal, parameters, adaptive_margin=False)
    adaptive = replay_episode(frames, start, goal, parameters, adaptive_margin=True)
    timing = benchmark_callable(
        lambda: choose_reactive_velocity(
            start,
            goal,
            positions,
            np.zeros_like(positions),
            parameters.fixed_clearance_m,
            parameters,
        ),
        warmup=1,
        repeats=3,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "fixed_steps": fixed.steps,
                "adaptive_steps": adaptive.steps,
                "planner_median_ms": timing["median_ms"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--latency-warmup", type=int, default=10)
    parser.add_argument("--latency-repeats", type=int, default=100)
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
