# Author: Amir Ghorbani
"""Generate leakage-resistant member-level predictions for referee-3 analyses.

This script deliberately stops before model comparison.  It builds a causal
0.8/0.4-s task with a past-velocity axis, a past-only speed-quality screen,
wall-free graph inputs, and every chronological evaluation origin retained.
The expensive graph fits are cached one member at a time so interrupted runs
can resume without changing a seed.  All outputs use the ``referee3``
namespace; frozen ``referee2`` evidence is never overwritten.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any
import zipfile

import numpy as np
from sklearn.covariance import LedoitWolf

from causal_anchor_revision import anchor, settings
from pedgeom.benchmarks import fit_gradient_boosted_residual, invariant_design
from pedgeom.calibration import VelocitySamples, build_velocity_samples
from pedgeom.datasets import load_julich_crossing_trajectory, load_julich_trajectory
from pedgeom.graph_baseline import fit_graph_interaction_network


_WORKER_TRAIN: list[VelocitySamples] | None = None
_WORKER_EVALUATION: list[VelocitySamples] | None = None
_WORKER_OPTIONS: dict[str, Any] | None = None
_CACHE_SCHEMA_VERSION = 1
_BUNDLE_SCHEMA_VERSION = 1


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_savez(path: Path, **arrays: Any) -> None:
    """Write a NumPy archive atomically so interrupted fits cannot look complete."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp.npz"
    )
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _update_array_digest(digest: Any, name: str, values: np.ndarray | None) -> None:
    digest.update(name.encode("utf-8"))
    if values is None:
        digest.update(b"<none>")
        return
    array = np.asarray(values)
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    if array.dtype.hasobject:
        digest.update(json.dumps(array.tolist(), default=str).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))


def _batch_fingerprint(batch: VelocitySamples) -> str:
    """Hash every field that can affect fitting, prediction order, or bundling."""

    digest = hashlib.sha256()
    digest.update(batch.run.encode("utf-8"))
    digest.update(str(batch.goal_method).encode("utf-8"))
    for name, values in (
        ("features", batch.features),
        ("target", batch.target),
        ("pedestrian_id", batch.pedestrian_id),
        ("frame", batch.frame),
        ("origin_time_s", batch.origin_time_s),
        ("raw_neighbours", batch.raw_neighbours),
    ):
        _update_array_digest(digest, name, values)
    return digest.hexdigest()


def _pool_cache_signature(
    training: list[VelocitySamples],
    evaluation: list[VelocitySamples],
    options: dict[str, Any],
) -> str:
    payload = {
        "schema": _CACHE_SCHEMA_VERSION,
        "options": options,
        "training": [_batch_fingerprint(batch) for batch in training],
        "evaluation": [_batch_fingerprint(batch) for batch in evaluation],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _worker_initialise(
    training: list[VelocitySamples],
    evaluation: list[VelocitySamples],
    options: dict[str, Any],
) -> None:
    global _WORKER_TRAIN, _WORKER_EVALUATION, _WORKER_OPTIONS
    _WORKER_TRAIN = training
    _WORKER_EVALUATION = evaluation
    _WORKER_OPTIONS = options


def _stage_worker_batch(directory: Path, prefix: str, batch: VelocitySamples) -> dict[str, Any]:
    """Store only graph-required arrays as mmap-compatible ``.npy`` files."""

    if batch.raw_neighbours is None:
        raise ValueError(f"{batch.run} lacks raw-neighbour states required by the graph")
    arrays = {
        "features": np.asarray(batch.features),
        "target": np.asarray(batch.target),
        "pedestrian_id": np.asarray(batch.pedestrian_id),
        "frame": np.asarray(batch.frame),
        "raw_neighbours": np.asarray(batch.raw_neighbours),
    }
    files: dict[str, str] = {}
    for name, values in arrays.items():
        if values.dtype.hasobject:
            if name != "pedestrian_id":
                raise ValueError(f"worker array {name} for {batch.run} has object dtype")
            values = values.astype(str)
        file_name = f"{prefix}_{name}.npy"
        np.save(directory / file_name, np.ascontiguousarray(values), allow_pickle=False)
        files[name] = file_name
    return {
        "run": batch.run,
        "goal_method": batch.goal_method,
        "files": files,
    }


def _stage_worker_payload(
    directory: Path,
    training: list[VelocitySamples],
    evaluation: list[VelocitySamples],
) -> Path:
    """Create a small manifest pointing to shared, read-only worker arrays."""

    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "training": [
            _stage_worker_batch(directory, f"training_{index:03d}", batch)
            for index, batch in enumerate(training)
        ],
        "evaluation": [
            _stage_worker_batch(directory, f"evaluation_{index:03d}", batch)
            for index, batch in enumerate(evaluation)
        ],
    }
    manifest = directory / "payload.json"
    _atomic_write_json(manifest, payload)
    return manifest


def _load_worker_batch(directory: Path, record: dict[str, Any]) -> VelocitySamples:
    files = record["files"]

    def load(name: str) -> np.ndarray:
        return np.load(directory / files[name], mmap_mode="r", allow_pickle=False)

    return VelocitySamples(
        run=str(record["run"]),
        features=load("features"),
        target=load("target"),
        pedestrian_id=load("pedestrian_id"),
        frame=load("frame"),
        goal_method=str(record["goal_method"]),
        raw_neighbours=load("raw_neighbours"),
    )


def _worker_initialise_file_backed(
    manifest_path: str,
    options: dict[str, Any],
) -> None:
    """Initialise a spawned worker from shared mmap files, not pickled batches."""

    manifest = Path(manifest_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    directory = manifest.parent
    training = [_load_worker_batch(directory, row) for row in payload["training"]]
    evaluation = [_load_worker_batch(directory, row) for row in payload["evaluation"]]
    _worker_initialise(training, evaluation, options)


def _fit_member_and_predict(seed: int) -> tuple[int, list[np.ndarray], float, int]:
    if _WORKER_TRAIN is None or _WORKER_EVALUATION is None or _WORKER_OPTIONS is None:
        raise RuntimeError("worker was not initialised")
    started = time.perf_counter()
    model = fit_graph_interaction_network(
        _WORKER_TRAIN,
        width=int(_WORKER_OPTIONS["width"]),
        epochs=int(_WORKER_OPTIONS["epochs"]),
        seed=seed,
        residual=bool(_WORKER_OPTIONS["residual"]),
        include_wall=bool(_WORKER_OPTIONS["include_wall"]),
        num_threads=int(_WORKER_OPTIONS["threads_per_worker"]),
    )
    predictions = [
        model.predict(
            batch,
            speed_cap=float(_WORKER_OPTIONS["raw_prediction_speed_cap_mps"]),
        ).astype(np.float32)
        for batch in _WORKER_EVALUATION
    ]
    return seed, predictions, time.perf_counter() - started, model.parameter_count


def _wall_free_design(batch: VelocitySamples) -> np.ndarray:
    return np.delete(invariant_design(batch), (5, 11), axis=1)


def _fit_mahalanobis(training: list[VelocitySamples]) -> tuple[np.ndarray, np.ndarray]:
    design = np.concatenate([_wall_free_design(batch) for batch in training])
    fitted = LedoitWolf().fit(design)
    return fitted.location_, fitted.precision_


def _mahalanobis(
    batch: VelocitySamples, location: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    centred = _wall_free_design(batch) - location
    squared = np.einsum("ni,ij,nj->n", centred, precision, centred)
    return np.sqrt(np.maximum(squared, 0.0))


def _build_controlled(
    source: Path,
    *,
    crossing: bool,
    maximum_samples: int | None,
    seed: int,
    config: dict[str, Any],
) -> VelocitySamples:
    return build_velocity_samples(
        source,
        interaction_range=float(config["task"]["interaction_range_m"]),
        horizon_frames=10,
        history_frames=20,
        frame_stride=10,
        frames_per_second=25.0,
        maximum_samples=maximum_samples,
        radius=float(config["task"]["pedestrian_radius_m"]),
        wall_range=0.25,
        wall_y_bounds=None,
        neighbour_cutoff=float(config["task"]["neighbour_cutoff_m"]),
        goal_method="prior_velocity",
        cardinal_routes=False,
        speed_outlier_threshold=None,
        past_speed_outlier_threshold=float(
            config["task"]["past_speed_quality_limit_mps"]
        ),
        target_speed_outlier_threshold=None,
        filter_target_speed=False,
        loader=(load_julich_crossing_trajectory if crossing else load_julich_trajectory),
        seed=seed,
        maximum_neighbours=int(config["task"]["maximum_neighbours"]),
    )


def build_julich_batches(
    project: Path, config: dict[str, Any]
) -> tuple[
    list[VelocitySamples],
    list[VelocitySamples],
    list[tuple[str, VelocitySamples]],
]:
    splits = json.loads((project / "data/splits.json").read_text(encoding="utf-8"))
    corridor = project / "data/raw/2013bidirectional"
    crossing = project / "data/raw/2013crossing90/trajectories"
    base_seed = int(config["replication"]["member_seed_base"])
    cap = int(config["task"]["training_samples_per_run"])

    training_fit: list[VelocitySamples] = []
    calibration_full: list[VelocitySamples] = []
    for index, run in enumerate(splits["calibration"]):
        path = corridor / f"{run}.txt"
        calibration_full.append(
            _build_controlled(
                path,
                crossing=False,
                maximum_samples=None,
                seed=base_seed + index,
                config=config,
            )
        )
        training_fit.append(
            _build_controlled(
                path,
                crossing=False,
                maximum_samples=cap,
                seed=base_seed + index,
                config=config,
            )
        )

    evaluation: list[tuple[str, VelocitySamples]] = []
    roles = (
        ("familiar corridor", "held_out_validation"),
        ("altered geometry", "held_out_geometry_stress"),
    )
    for role, split_name in roles:
        for index, run in enumerate(splits[split_name]):
            evaluation.append(
                (
                    role,
                    _build_controlled(
                        corridor / f"{run}.txt",
                        crossing=False,
                        maximum_samples=None,
                        seed=base_seed + 100 + index,
                        config=config,
                    ),
                )
            )
    for index, path in enumerate(sorted(crossing.glob("crossing_90_[de]_*.txt"))):
        evaluation.append(
            (
                "crossing topology",
                _build_controlled(
                    path,
                    crossing=True,
                    maximum_samples=None,
                    seed=base_seed + 200 + index,
                    config=config,
                ),
            )
        )
    return training_fit, calibration_full, evaluation


def _metadata(batch: VelocitySamples) -> dict[str, np.ndarray]:
    if batch.origin_time_s is None:
        raise ValueError(f"{batch.run} does not contain explicit origin times")
    if batch.position is None or batch.neighbour_count is None or batch.occupancy is None:
        raise ValueError(f"{batch.run} lacks required diagnostics")
    metadata = {
        "target": np.asarray(batch.target, dtype=np.float32),
        "frame": np.asarray(batch.frame, dtype=np.int64),
        "origin_time_s": np.asarray(batch.origin_time_s, dtype=np.float64),
        "pedestrian_id": np.asarray(batch.pedestrian_id),
        "position": np.asarray(batch.position, dtype=np.float32),
        "prior_velocity": np.asarray(batch.features[:, 0], dtype=np.float32),
        "neighbour_count": np.asarray(batch.neighbour_count, dtype=np.float32),
        "occupancy": np.asarray(batch.occupancy, dtype=np.float32),
    }
    if metadata["pedestrian_id"].dtype.hasobject:
        metadata["pedestrian_id"] = metadata["pedestrian_id"].astype(str)
    sample_count = len(batch.target)
    expected_shapes = {
        "target": (sample_count, 2),
        "frame": (sample_count,),
        "origin_time_s": (sample_count,),
        "pedestrian_id": (sample_count,),
        "position": (sample_count, 2),
        "prior_velocity": (sample_count, 2),
        "neighbour_count": (sample_count,),
        "occupancy": (sample_count,),
    }
    for name, expected in expected_shapes.items():
        if metadata[name].shape != expected:
            raise ValueError(
                f"metadata {name} for {batch.run} must have shape {expected}, "
                f"got {metadata[name].shape}"
            )
    for name in (
        "target",
        "origin_time_s",
        "position",
        "prior_velocity",
        "neighbour_count",
        "occupancy",
    ):
        if not np.all(np.isfinite(metadata[name])):
            raise ValueError(f"metadata {name} contains non-finite values for {batch.run}")
    if np.any(np.diff(metadata["origin_time_s"]) < 0.0):
        raise ValueError(f"origin times are not chronological for {batch.run}")
    return metadata


def _write_bundle(
    path: Path,
    batch: VelocitySamples,
    members: np.ndarray,
    fallback: np.ndarray,
    constant_velocity: np.ndarray,
    mahalanobis: np.ndarray,
    *,
    member_seeds: list[int],
    role: str,
    fit_role: str,
) -> None:
    members = np.asarray(members, dtype=np.float32)
    fallback = np.asarray(fallback, dtype=np.float32)
    constant_velocity = np.asarray(constant_velocity, dtype=np.float32)
    mahalanobis = np.asarray(mahalanobis, dtype=np.float32)
    sample_count = len(batch.target)
    expected_vector_shape = (sample_count, 2)
    if members.ndim != 3 or members.shape[1:] != expected_vector_shape:
        raise ValueError(
            f"member predictions for {batch.run} must have shape "
            f"(M, {sample_count}, 2), got {members.shape}"
        )
    if len(member_seeds) != members.shape[0]:
        raise ValueError("member_seeds must identify every member prediction")
    for name, values in (
        ("fallback", fallback),
        ("constant_velocity", constant_velocity),
    ):
        if values.shape != expected_vector_shape:
            raise ValueError(
                f"{name} predictions for {batch.run} must have shape "
                f"{expected_vector_shape}, got {values.shape}"
            )
    if mahalanobis.shape != (sample_count,):
        raise ValueError(
            f"Mahalanobis scores for {batch.run} must have shape "
            f"({sample_count},), got {mahalanobis.shape}"
        )
    for name, values in (
        ("members", members),
        ("fallback", fallback),
        ("constant_velocity", constant_velocity),
        ("mahalanobis", mahalanobis),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values for {batch.run}")

    _atomic_savez(
        path,
        bundle_schema_version=np.asarray(_BUNDLE_SCHEMA_VERSION, dtype=np.int64),
        **_metadata(batch),
        members_raw=members,
        member_seeds=np.asarray(member_seeds, dtype=np.int64),
        fallback_raw=fallback,
        constant_velocity=constant_velocity,
        mahalanobis=mahalanobis,
        run=np.asarray(batch.run),
        role=np.asarray(role),
        fit_role=np.asarray(fit_role),
        goal_method=np.asarray(batch.goal_method),
    )


def _member_seeds(config: dict[str, Any], pool_size: int | None = None) -> list[int]:
    size = int(config["replication"]["member_pool_size"] if pool_size is None else pool_size)
    base = int(config["replication"]["member_seed_base"])
    stride = int(config["replication"]["member_seed_stride"])
    if size < 1:
        raise ValueError("member pool size must be at least one")
    seeds = [base + stride * index for index in range(size)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("member seeds must be unique")
    return seeds


def _member_cache_path(cache_prefix: Path, seed: int) -> Path:
    return cache_prefix.with_name(f"{cache_prefix.name}_{seed}.npz")


def _write_member_cache(
    path: Path,
    *,
    seed: int,
    predictions: list[np.ndarray],
    seconds: float,
    parameters: int,
    options: dict[str, Any],
    signature: str,
    evaluation: list[VelocitySamples],
) -> None:
    if len(predictions) != len(evaluation):
        raise ValueError("worker returned the wrong number of evaluation predictions")
    arrays: dict[str, Any] = {
        "cache_schema_version": np.asarray(_CACHE_SCHEMA_VERSION, dtype=np.int64),
        "cache_signature": np.asarray(signature),
        "seed": np.asarray(seed, dtype=np.int64),
        "prediction_count": np.asarray(len(predictions), dtype=np.int64),
        "run_names": np.asarray([batch.run for batch in evaluation]),
        "run_lengths": np.asarray([len(batch.target) for batch in evaluation], dtype=np.int64),
        "seconds": np.asarray(seconds, dtype=np.float64),
        "parameters": np.asarray(parameters, dtype=np.int64),
        "epochs": np.asarray(options["epochs"], dtype=np.int64),
        "width": np.asarray(options["width"], dtype=np.int64),
        "wall_features": np.asarray(False),
    }
    for index, (prediction, batch) in enumerate(zip(predictions, evaluation, strict=True)):
        values = np.asarray(prediction, dtype=np.float32)
        if values.shape != (len(batch.target), 2):
            raise ValueError(
                f"prediction {index} for {batch.run} has shape {values.shape}, "
                f"expected {(len(batch.target), 2)}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"prediction {index} for {batch.run} is non-finite")
        arrays[f"prediction_{index}"] = values
    _atomic_savez(path, **arrays)


def _load_member_cache(
    path: Path,
    *,
    seed: int,
    signature: str,
    evaluation: list[VelocitySamples],
) -> tuple[list[np.ndarray], dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["cache_schema_version"].item()) != _CACHE_SCHEMA_VERSION:
                return None
            if str(payload["cache_signature"].item()) != signature:
                return None
            if int(payload["seed"].item()) != seed:
                return None
            if int(payload["prediction_count"].item()) != len(evaluation):
                return None
            expected_names = np.asarray([batch.run for batch in evaluation])
            expected_lengths = np.asarray(
                [len(batch.target) for batch in evaluation], dtype=np.int64
            )
            if not np.array_equal(payload["run_names"], expected_names):
                return None
            if not np.array_equal(payload["run_lengths"], expected_lengths):
                return None
            predictions: list[np.ndarray] = []
            for index, batch in enumerate(evaluation):
                values = np.asarray(payload[f"prediction_{index}"], dtype=np.float32)
                if values.shape != (len(batch.target), 2) or not np.all(np.isfinite(values)):
                    return None
                predictions.append(np.array(values, copy=True))
            metadata = {
                "seconds": float(payload["seconds"].item()),
                "parameters": int(payload["parameters"].item()),
                "epochs": int(payload["epochs"].item()),
                "width": int(payload["width"].item()),
                "wall_features": bool(payload["wall_features"].item()),
            }
    except (KeyError, OSError, ValueError, EOFError, zipfile.BadZipFile):
        return None
    return predictions, metadata


def _predict_member_pool(
    training: list[VelocitySamples],
    evaluation: list[VelocitySamples],
    seeds: list[int],
    *,
    cache_prefix: Path,
    config: dict[str, Any],
    workers: int,
    manifest_rows: list[dict[str, Any]],
) -> list[np.ndarray]:
    if workers < 1:
        raise ValueError("workers must be at least one")
    if not training:
        raise ValueError("training must contain at least one complete run")
    if not evaluation:
        raise ValueError("evaluation must contain at least one complete run")
    if not seeds:
        raise ValueError("seeds must contain at least one member seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError("member seeds must be unique")
    cache_prefix.parent.mkdir(parents=True, exist_ok=True)
    # A single thread per fit makes cache contents independent of the requested
    # process count and prevents CPU oversubscription on Windows.
    options = {
        "width": int(config["replication"]["graph_width"]),
        "epochs": int(config["replication"]["training_epochs"]),
        "threads_per_worker": 1,
        "residual": True,
        "include_wall": False,
        "raw_prediction_speed_cap_mps": 100.0,
        "graph_implementation_sha256": _sha256(
            project_root() / "src/pedgeom/graph_baseline.py"
        ),
    }
    signature = _pool_cache_signature(training, evaluation, options)
    loaded: dict[int, tuple[list[np.ndarray], dict[str, Any]]] = {}
    missing: list[int] = []
    for seed in seeds:
        cache_path = _member_cache_path(cache_prefix, seed)
        cached = _load_member_cache(
            cache_path,
            seed=seed,
            signature=signature,
            evaluation=evaluation,
        )
        if cached is None:
            missing.append(seed)
        else:
            loaded[seed] = cached
    if missing:
        results: list[tuple[int, list[np.ndarray], float, int]] = []
        if workers == 1:
            _worker_initialise(training, evaluation, options)
            results = [_fit_member_and_predict(seed) for seed in missing]
        else:
            # Windows uses spawn: passing VelocitySamples in ``initargs`` would
            # pickle the full training and evaluation arrays once per process.
            # Stage graph-required arrays once and let every worker mmap them.
            with TemporaryDirectory(
                prefix=f".{cache_prefix.name}_workers_",
                dir=cache_prefix.parent,
            ) as temporary:
                manifest = _stage_worker_payload(
                    Path(temporary), training, evaluation
                )
                with ProcessPoolExecutor(
                    max_workers=min(workers, len(missing)),
                    initializer=_worker_initialise_file_backed,
                    initargs=(str(manifest), options),
                ) as executor:
                    futures = {
                        executor.submit(_fit_member_and_predict, seed): seed
                        for seed in missing
                    }
                    for future in as_completed(futures):
                        results.append(future.result())
        for seed, predictions, seconds, parameters in results:
            cache_path = _member_cache_path(cache_prefix, seed)
            _write_member_cache(
                cache_path,
                seed=seed,
                predictions=predictions,
                seconds=seconds,
                parameters=parameters,
                options=options,
                signature=signature,
                evaluation=evaluation,
            )
            cached = _load_member_cache(
                cache_path,
                seed=seed,
                signature=signature,
                evaluation=evaluation,
            )
            if cached is None:
                raise RuntimeError(f"failed to verify member cache {cache_path}")
            loaded[seed] = cached
            print(
                f"completed {cache_prefix.name}, seed {seed}, {seconds:.1f} s",
                flush=True,
            )
    by_member: list[list[np.ndarray]] = []
    for seed in seeds:
        cache_path = _member_cache_path(cache_prefix, seed)
        predictions, metadata = loaded[seed]
        by_member.append(predictions)
        manifest_rows.append(
            {
                "cache": cache_prefix.name,
                "cache_file": cache_path.name,
                "cache_sha256": _sha256(cache_path),
                "cache_signature": signature,
                "seed": seed,
                **metadata,
            }
        )
    # Runs have different lengths and therefore cannot be stacked along the
    # evaluation axis. Return one M x N x 2 array for each complete run.
    per_run = [
        np.stack([member[index] for member in by_member], axis=0)
        for index in range(len(evaluation))
    ]
    return per_run


def _fit_non_neural(
    training: list[VelocitySamples],
    evaluation: list[VelocitySamples],
    config: dict[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    project = project_root()
    _, hgb_settings, _ = settings(project)
    seed = int(config["replication"]["member_seed_base"])
    fallback = fit_gradient_boosted_residual(
        training,
        anchor(),
        seed=seed,
        **hgb_settings,
    )
    location, precision = _fit_mahalanobis(training)
    fallback_speed_cap = float(
        config["experts"]["fallback_prediction_speed_cap_mps"]
    )
    fallback_predictions = [
        fallback.predict(batch, speed_cap=fallback_speed_cap) for batch in evaluation
    ]
    cv_predictions = [anchor().predict(batch, speed_cap=100.0) for batch in evaluation]
    mahalanobis = [_mahalanobis(batch, location, precision) for batch in evaluation]
    return fallback_predictions, cv_predictions, mahalanobis


def generate_julich(args: argparse.Namespace) -> None:
    project = project_root()
    config_path = project / "configs/referee3_lock.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    processed = project / "data/processed"
    bundle_dir = processed / "referee3_predictions"
    cache_dir = processed / "referee3_member_cache"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    training, calibration_full, evaluation = build_julich_batches(project, config)
    seeds = _member_seeds(config, args.pool_size)
    manifest_rows: list[dict[str, Any]] = []
    bundle_rows: list[dict[str, Any]] = []

    for held_index, held_batch in enumerate(calibration_full):
        fold_training = [
            batch for index, batch in enumerate(training) if index != held_index
        ]
        fold_seeds = [seed + 100_000 * (held_index + 1) for seed in seeds]
        member_by_run = _predict_member_pool(
            fold_training,
            [held_batch],
            fold_seeds,
            cache_prefix=cache_dir / f"oof_fold_{held_index:02d}",
            config=config,
            workers=args.workers,
            manifest_rows=manifest_rows,
        )
        fallback, cv, mahalanobis = _fit_non_neural(
            fold_training, [held_batch], config
        )
        bundle_path = bundle_dir / f"oof_{held_index:02d}_{held_batch.run}.npz"
        _write_bundle(
            bundle_path,
            held_batch,
            member_by_run[0],
            fallback[0],
            cv[0],
            mahalanobis[0],
            member_seeds=fold_seeds,
            role="calibration",
            fit_role="leave-one-complete-run-out",
        )
        bundle_rows.append(
            {
                "path": str(bundle_path.relative_to(project)),
                "run": held_batch.run,
                "role": "calibration",
                "samples": len(held_batch.target),
                "members": len(fold_seeds),
                "member_seeds": fold_seeds,
                "sha256": _sha256(bundle_path),
            }
        )
        print(f"wrote OOF bundle {held_batch.run}", flush=True)

    evaluation_batches = [batch for _, batch in evaluation]
    member_by_run = _predict_member_pool(
        training,
        evaluation_batches,
        seeds,
        cache_prefix=cache_dir / "deployed_all_calibration_runs",
        config=config,
        workers=args.workers,
        manifest_rows=manifest_rows,
    )
    fallback, cv, mahalanobis = _fit_non_neural(training, evaluation_batches, config)
    for index, (role, batch) in enumerate(evaluation):
        safe_role = role.replace(" ", "_")
        bundle_path = bundle_dir / f"test_{safe_role}_{batch.run}.npz"
        _write_bundle(
            bundle_path,
            batch,
            member_by_run[index],
            fallback[index],
            cv[index],
            mahalanobis[index],
            member_seeds=seeds,
            role=role,
            fit_role="all calibration runs",
        )
        bundle_rows.append(
            {
                "path": str(bundle_path.relative_to(project)),
                "run": batch.run,
                "role": role,
                "samples": len(batch.target),
                "members": len(seeds),
                "member_seeds": seeds,
                "sha256": _sha256(bundle_path),
            }
        )
        print(f"wrote test bundle {role}: {batch.run}", flush=True)

    manifest_path = processed / "referee3_training_manifest.json"
    by_key = {(row["cache"], row["seed"]): row for row in manifest_rows}
    payload = {
        "manifest_schema_version": 1,
        "config_sha256": _sha256(config_path),
        "input_sha256": {
            "splits": _sha256(project / "data/splits.json"),
            "fallback_settings": _sha256(
                project / "data/processed/prospective_selection.json"
            ),
        },
        "implementation_sha256": {
            "generator": _sha256(Path(__file__)),
            "graph": _sha256(project / "src/pedgeom/graph_baseline.py"),
            "calibration": _sha256(project / "src/pedgeom/calibration.py"),
            "datasets": _sha256(project / "src/pedgeom/datasets.py"),
            "benchmarks": _sha256(project / "src/pedgeom/benchmarks.py"),
            "fallback_settings_loader": _sha256(
                project / "experiments/causal_anchor_revision.py"
            ),
        },
        "pool_size": len(seeds),
        "seeds": seeds,
        "fits": sorted(by_key.values(), key=lambda row: (row["cache"], row["seed"])),
        "bundles": sorted(bundle_rows, key=lambda row: row["path"]),
    }
    _atomic_write_json(manifest_path, payload)
    print(f"wrote {manifest_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pool-size", type=int)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least one")
    if args.pool_size is not None and args.pool_size < 1:
        parser.error("--pool-size must be at least one")
    generate_julich(args)


if __name__ == "__main__":
    main()
