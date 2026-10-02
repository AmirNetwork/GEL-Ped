# Author: Amir Ghorbani
"""Generate the frozen one-shot T-junction prediction bundles without analysis.

This program validates the exact registered archive before parsing, rebuilds
the final 25-member graph pool from the seven frozen calibration runs, and
writes prediction bundles for all registered T-junction runs.  Held-out target
values are replaced by zeros in every predictor-facing batch.  No error,
comparison, gate-performance, or tuning statistic is computed here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZipFile

import numpy as np

from pedgeom.calibration import VelocitySamples, build_velocity_samples
from pedgeom.datasets import load_resampled_julich_archive_trajectory
from referee3_generate_predictions import (
    _atomic_write_json,
    _batch_fingerprint,
    _fit_non_neural,
    _member_seeds,
    _predict_member_pool,
    _sha256,
    _target_blind_prediction_view,
    _write_bundle,
    build_julich_deployed_sentinel_batch,
    build_julich_fit_batches,
)


DEFAULT_WORK_DIRECTORY = Path(r"E:\GEL-Ped-TRC-Review-Compute")
DEFAULT_ARCHIVE = (
    DEFAULT_WORK_DIRECTORY
    / "tjunction"
    / "2009tjunction_trajectories_txt.zip"
)
FINAL_DIRECTORY_NAME = "referee3_tjunction_predictions"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _exact_frame_count(seconds: float, frames_per_second: float) -> int:
    value = seconds * frames_per_second
    rounded = int(round(value))
    if not np.isclose(value, rounded, atol=1e-12, rtol=0.0):
        raise ValueError(
            f"{seconds} s is not an integer number of frames at "
            f"{frames_per_second} Hz"
        )
    return rounded


def validate_frozen_config(
    config_path: Path,
    registry: dict[str, Any],
) -> dict[str, Any]:
    actual_hash = _sha256(config_path)
    expected_hash = str(registry["referee3_config_sha256"]).lower()
    if actual_hash.lower() != expected_hash:
        raise RuntimeError(
            "referee3 configuration differs from the pre-heldout registry: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    config = _load_json(config_path)
    if int(config["replication"]["member_pool_size"]) != 25:
        raise RuntimeError("the heldout path requires the frozen 25-member pool")
    if bool(config["task"]["target_speed_filter"]):
        raise RuntimeError("heldout membership must not use future target speed")
    return config


def inspect_registered_archive(
    archive_path: Path,
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Validate checksum, members, byte metadata, and schema without outcomes."""

    if not archive_path.is_file():
        raise FileNotFoundError(f"registered heldout archive not found: {archive_path}")
    archive_hash = _sha256(archive_path)
    expected_hash = str(registry["archive_sha256"]).lower()
    if archive_hash.lower() != expected_hash:
        raise RuntimeError(
            f"heldout archive SHA256 mismatch: expected {expected_hash}, "
            f"got {archive_hash}"
        )
    archive_bytes = archive_path.stat().st_size
    if archive_bytes != int(registry["archive_bytes"]):
        raise RuntimeError("heldout archive byte size differs from the registry")

    registered = {row["name"]: row for row in registry["members"]}
    inventory_rows: list[dict[str, Any]] = []
    with ZipFile(archive_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != list(registered):
            raise RuntimeError(
                "heldout archive member order/set differs from the frozen registry"
            )
        for info in infos:
            expected = registered[info.filename]
            if info.is_dir() or Path(info.filename).name != info.filename:
                raise RuntimeError("heldout archive must contain only flat files")
            if (
                info.file_size != int(expected["bytes"])
                or info.compress_size != int(expected["compressed_bytes"])
                or f"{info.CRC:08x}" != str(expected["crc32"]).lower()
            ):
                raise RuntimeError(
                    f"ZIP metadata differs from registry for {info.filename}"
                )

            rows = 0
            blank_rows = 0
            comment_rows = 0
            with archive.open(info) as stream:
                for raw_line in stream:
                    stripped = raw_line.strip()
                    if not stripped:
                        blank_rows += 1
                        continue
                    if stripped.startswith(b"#"):
                        comment_rows += 1
                        continue
                    fields = stripped.split()
                    if len(fields) != int(expected["columns"]):
                        raise RuntimeError(
                            f"wrong column count in {info.filename} row {rows + 1}"
                        )
                    try:
                        int(fields[0])
                        int(fields[1])
                        numeric = np.asarray(
                            [float(value) for value in fields[2:]], dtype=float
                        )
                    except (TypeError, ValueError) as error:
                        raise RuntimeError(
                            f"non-numeric schema in {info.filename} row {rows + 1}"
                        ) from error
                    if not np.all(np.isfinite(numeric)):
                        raise RuntimeError(
                            f"non-finite schema in {info.filename} row {rows + 1}"
                        )
                    rows += 1
            if rows != int(expected["rows"]):
                raise RuntimeError(f"row count differs from registry for {info.filename}")
            if blank_rows or comment_rows:
                raise RuntimeError(
                    f"unexpected blank/comment rows in {info.filename}"
                )
            inventory_rows.append(
                {
                    "name": info.filename,
                    "bytes": info.file_size,
                    "compressed_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                    "rows": rows,
                    "columns": int(expected["columns"]),
                }
            )

    return {
        "inventory_schema_version": 1,
        "analysis_freeze_commit": registry["analysis_freeze_commit"],
        "source_url": registry["source_url"],
        "archive_name": archive_path.name,
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_hash,
        "trajectory_schema": registry["trajectory_schema"],
        "source_fps": registry["source_fps"],
        "target_fps": registry["target_fps"],
        "resampling": registry["resampling"],
        "members": inventory_rows,
        "inspection_scope": "archive structure and schema only",
        "performance_analysis_run": False,
    }


def write_inventory_once(path: Path, inventory: dict[str, Any]) -> None:
    if path.exists():
        existing = _load_json(path)
        if existing != inventory:
            raise RuntimeError(f"existing inventory is non-identical: {path}")
        return
    _atomic_write_json(path, inventory)


def _build_heldout_batch(
    archive_path: Path,
    member: str,
    *,
    config: dict[str, Any],
    registry: dict[str, Any],
    seed: int,
) -> VelocitySamples:
    source_fps = float(registry["source_fps"])
    target_fps = float(registry["target_fps"])
    history_frames = _exact_frame_count(
        float(config["task"]["history_seconds"]), target_fps
    )
    horizon_frames = _exact_frame_count(
        float(config["task"]["prediction_seconds"]), target_fps
    )
    frame_stride = _exact_frame_count(
        float(config["task"]["origin_spacing_seconds"]), target_fps
    )

    def loader(_unused: Path):
        return load_resampled_julich_archive_trajectory(
            archive_path,
            member,
            source_fps=source_fps,
            target_fps=target_fps,
        )

    batch = build_velocity_samples(
        Path(member),
        interaction_range=float(config["task"]["interaction_range_m"]),
        horizon_frames=horizon_frames,
        history_frames=history_frames,
        frame_stride=frame_stride,
        frames_per_second=target_fps,
        maximum_samples=None,
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
        loader=loader,
        seed=seed,
        maximum_neighbours=int(config["task"]["maximum_neighbours"]),
    )
    if len(batch.target) == 0:
        raise RuntimeError(f"registered heldout run has no eligible origins: {member}")
    if batch.origin_time_s is None or not np.allclose(
        batch.origin_time_s,
        batch.frame / target_fps,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(f"invalid exact time grid for {member}")
    return batch


def build_heldout_batches(
    archive_path: Path,
    config: dict[str, Any],
    registry: dict[str, Any],
) -> list[VelocitySamples]:
    base_seed = int(config["replication"]["member_seed_base"])
    return [
        _build_heldout_batch(
            archive_path,
            row["name"],
            config=config,
            registry=registry,
            seed=base_seed + 500 + index,
        )
        for index, row in enumerate(registry["members"])
    ]


def _assert_one_shot_available(work_directory: Path) -> Path:
    final_directory = work_directory / FINAL_DIRECTORY_NAME
    if final_directory.exists():
        raise RuntimeError(
            "one-shot heldout prediction directory already exists; refusing to overwrite "
            f"{final_directory}"
        )
    return final_directory


def verify_deployed_sentinel(
    sentinel_path: Path,
    sentinel_batch: VelocitySamples,
    seeds: list[int],
    *,
    config_path: Path,
    graph_path: Path,
) -> np.ndarray:
    """Validate the frozen Jülich sentinel before any heldout result is accepted."""

    if not sentinel_path.is_file():
        raise FileNotFoundError(f"deployed graph sentinel not found: {sentinel_path}")
    try:
        with np.load(sentinel_path, allow_pickle=False) as payload:
            if int(payload["sentinel_schema_version"].item()) != 1:
                raise RuntimeError("unsupported deployed sentinel schema")
            if str(payload["run"].item()) != sentinel_batch.run:
                raise RuntimeError("deployed sentinel run differs from reconstruction")
            fingerprint = _batch_fingerprint(
                _target_blind_prediction_view(sentinel_batch)
            )
            if str(payload["batch_fingerprint"].item()) != fingerprint:
                raise RuntimeError("deployed sentinel input fingerprint mismatch")
            if str(payload["config_sha256"].item()) != _sha256(config_path):
                raise RuntimeError("deployed sentinel config hash mismatch")
            if str(payload["graph_implementation_sha256"].item()) != _sha256(
                graph_path
            ):
                raise RuntimeError("deployed sentinel graph hash mismatch")
            if bool(payload["target_labels_included"].item()):
                raise RuntimeError("deployed sentinel must not contain target labels")
            if not np.array_equal(payload["seeds"], np.asarray(seeds, dtype=np.int64)):
                raise RuntimeError("deployed sentinel seeds differ from frozen pool")
            sample_count = int(payload["sample_count"].item())
            predictions = np.asarray(payload["predictions"], dtype=np.float32)
    except (KeyError, OSError, ValueError) as error:
        raise RuntimeError(f"invalid deployed graph sentinel: {sentinel_path}") from error
    if sample_count < 1 or predictions.shape != (len(seeds), sample_count, 2):
        raise RuntimeError("deployed sentinel prediction shape is invalid")
    if sample_count > len(sentinel_batch.target) or not np.all(np.isfinite(predictions)):
        raise RuntimeError("deployed sentinel predictions are invalid")
    return predictions


def execute_once(
    archive_path: Path,
    work_directory: Path,
    *,
    workers: int,
    sentinel_path: Path | None = None,
) -> Path:
    if workers < 1:
        raise ValueError("workers must be at least one")
    project = project_root()
    registry_path = project / "configs/referee3_heldout_registry.json"
    config_path = project / "configs/referee3_lock.json"
    registry = _load_json(registry_path)
    config = validate_frozen_config(config_path, registry)
    inventory = inspect_registered_archive(archive_path, registry)
    work_directory.mkdir(parents=True, exist_ok=True)
    write_inventory_once(
        work_directory / "tjunction" / "archive_inventory.json",
        inventory,
    )
    final_directory = _assert_one_shot_available(work_directory)

    training = build_julich_fit_batches(project, config)
    heldout = build_heldout_batches(archive_path, config, registry)
    blind_heldout = [_target_blind_prediction_view(batch) for batch in heldout]
    seeds = _member_seeds(config)
    if len(seeds) != 25:
        raise RuntimeError("the one-shot path requires exactly 25 member seeds")
    sentinel_batch = build_julich_deployed_sentinel_batch(project, config)
    sentinel_path = (
        project / "data/processed/referee3_deployed_sentinel.npz"
        if sentinel_path is None
        else sentinel_path
    )
    frozen_sentinel = verify_deployed_sentinel(
        sentinel_path,
        sentinel_batch,
        seeds,
        config_path=config_path,
        graph_path=project / "src/pedgeom/graph_baseline.py",
    )

    manifest_rows: list[dict[str, Any]] = []
    members_by_run = _predict_member_pool(
        training,
        [sentinel_batch, *heldout],
        seeds,
        cache_prefix=(
            work_directory
            / "referee3_member_cache"
            / "heldout_frozen_final_pool"
        ),
        config=config,
        workers=workers,
        manifest_rows=manifest_rows,
        evaluation_target_blind=True,
    )
    sentinel_count = frozen_sentinel.shape[1]
    if not np.array_equal(
        members_by_run[0][:, :sentinel_count],
        frozen_sentinel,
    ):
        raise RuntimeError(
            "deterministic final-pool refit failed the frozen Jülich sentinel"
        )
    members_by_run = members_by_run[1:]
    fallback, constant_velocity, mahalanobis = _fit_non_neural(
        training,
        blind_heldout,
        config,
    )

    with TemporaryDirectory(
        prefix=".referee3_tjunction_stage_",
        dir=work_directory,
    ) as temporary:
        stage = Path(temporary)
        bundle_rows: list[dict[str, Any]] = []
        for index, batch in enumerate(heldout):
            bundle_path = stage / f"heldout_{batch.run}.npz"
            _write_bundle(
                bundle_path,
                batch,
                members_by_run[index],
                fallback[index],
                constant_velocity[index],
                mahalanobis[index],
                member_seeds=seeds,
                role="untouched T-junction",
                fit_role="all calibration runs; frozen 25-member final pool",
            )
            bundle_rows.append(
                {
                    "file": bundle_path.name,
                    "run": batch.run,
                    "samples": len(batch.target),
                    "members": len(seeds),
                    "member_seeds": seeds,
                    "sha256": _sha256(bundle_path),
                }
            )

        by_key = {(row["cache"], row["seed"]): row for row in manifest_rows}
        receipt = {
            "receipt_schema_version": 1,
            "analysis_freeze_commit": registry["analysis_freeze_commit"],
            "referee3_config_sha256": _sha256(config_path),
            "heldout_registry_sha256": _sha256(registry_path),
            "archive_sha256": inventory["archive_sha256"],
            "archive_members": [row["name"] for row in inventory["members"]],
            "deployed_sentinel_sha256": _sha256(sentinel_path),
            "deployed_sentinel_verified": True,
            "pool_size": len(seeds),
            "seeds": seeds,
            "fits": sorted(
                by_key.values(), key=lambda row: (row["cache"], row["seed"])
            ),
            "bundles": bundle_rows,
            "target_labels_used_for_prediction": False,
            "performance_analysis_run": False,
            "fallback_specification": registry["fallback_specification"],
            "implementation_sha256": {
                "heldout_generator": _sha256(Path(__file__)),
                "prediction_generator": _sha256(
                    project / "experiments/referee3_generate_predictions.py"
                ),
                "datasets": _sha256(project / "src/pedgeom/datasets.py"),
                "calibration": _sha256(project / "src/pedgeom/calibration.py"),
                "graph": _sha256(project / "src/pedgeom/graph_baseline.py"),
                "benchmarks": _sha256(project / "src/pedgeom/benchmarks.py"),
            },
        }
        _atomic_write_json(stage / "ONE_SHOT_PREDICTION_RECEIPT.json", receipt)
        # Revalidate the archive immediately before publishing the atomic result.
        if _sha256(archive_path) != inventory["archive_sha256"]:
            raise RuntimeError("heldout archive changed during prediction generation")
        os.replace(stage, final_directory)

    print(f"wrote prediction-only heldout bundles to {final_directory}", flush=True)
    print("no performance analysis was run", flush=True)
    return final_directory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--work-directory", type=Path, default=DEFAULT_WORK_DIRECTORY)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sentinel", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inventory-only", action="store_true")
    mode.add_argument("--execute-once", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be at least one")
    if args.inventory_only:
        project = project_root()
        registry = _load_json(project / "configs/referee3_heldout_registry.json")
        validate_frozen_config(project / "configs/referee3_lock.json", registry)
        inventory = inspect_registered_archive(args.archive, registry)
        output = args.work_directory / "tjunction" / "archive_inventory.json"
        write_inventory_once(output, inventory)
        print(f"wrote schema-only archive inventory to {output}", flush=True)
        return
    execute_once(
        args.archive,
        args.work_directory,
        workers=args.workers,
        sentinel_path=args.sentinel,
    )


if __name__ == "__main__":
    main()
