# Author: Amir Ghorbani
"""Synthetic-only tests for the frozen T-junction analysis firewall."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import referee3_analyse_heldout as analysis  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(
    path: Path,
    *,
    run: str,
    role: str,
    seeds: list[int],
    heldout: bool,
    shift: float = 0.0,
) -> int:
    n = 20
    target = np.column_stack((np.ones(n), np.zeros(n)))
    offsets = np.linspace(-0.04, 0.04, len(seeds))
    if heldout:
        offsets = 3.0 * offsets
    members = np.stack(
        [target + np.array([shift + offset, 0.0]) for offset in offsets],
        axis=0,
    ).astype(np.float32)
    fallback = target.astype(np.float32)
    values = {
        "bundle_schema_version": np.asarray(1, dtype=np.int64),
        "target": target,
        "frame": np.arange(n, dtype=np.int64) * 4,
        "origin_time_s": np.arange(n, dtype=float) * 0.4,
        "pedestrian_id": np.ones(n, dtype=np.int64),
        "position": np.zeros((n, 2), dtype=float),
        "prior_velocity": target.copy(),
        "neighbour_count": np.ones(n, dtype=float),
        "occupancy": np.ones(n, dtype=float),
        "members_raw": members,
        "member_seeds": np.asarray(seeds, dtype=np.int64),
        "fallback_raw": fallback,
        "constant_velocity": target.copy(),
        "mahalanobis": np.zeros(n, dtype=float),
        "run": np.asarray(run),
        "role": np.asarray(role),
        "fit_role": np.asarray(
            analysis.EXPECTED_FIT_ROLE if heldout else "leave-one-complete-run-out"
        ),
        "goal_method": np.asarray("prior_velocity"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)
    return n


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _synthetic_frozen_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project = tmp_path / "project"
    work = tmp_path / "work"
    predictions = work / "referee3_tjunction_predictions"
    output = project / "reports/generated"
    seeds = [101 + 17 * index for index in range(25)]
    config = {
        "experts": {"fallback_prediction_speed_cap_mps": 3.0},
        "guard": {"trailing_window_seconds": 1.6, "primary_shape": "exponential"},
        "controls": {"fixed_blend_grid_step": 0.1},
        "replication": {
            "member_seed_base": 101,
            "member_seed_stride": 17,
            "member_pool_size": 25,
            "primary_member_indices": [0, 1, 2, 3, 4],
            "primary_ensemble_size": 5,
        },
        "endpoints": {
            "bootstrap_replicates": 100,
            "bootstrap_seed": 99,
            "equivalence_absolute_margin_mps": 0.002,
            "equivalence_relative_margin": 0.01,
        },
        "heldout": {"dataset": "synthetic T-junction", "minimum_complete_runs": 2},
        "multiplicity": {
            "confirmatory_family": [
                "heldout_graph_vs_guard_run_rmse",
                "heldout_graph_vs_guard_q95",
                "heldout_graph_vs_guard_cvar95",
                "heldout_graph_vs_guard_error_gt_0.5",
            ],
            "adjustment": "Holm",
        },
    }
    config_path = project / "configs/referee3_lock.json"
    _write_json(config_path, config)
    config_hash = _sha256(config_path)

    archive = work / "tjunction/archive.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"synthetic registered archive")
    registry_members = [
        {
            "name": f"run-{index}.txt",
            "bytes": 10 + index,
            "compressed_bytes": 5 + index,
            "crc32": f"{index:08x}",
            "rows": 100 + index,
            "columns": 5,
        }
        for index in range(3)
    ]
    fallback_specification = {
        "type": "feature-based HGB residual",
        "include_wall": False,
        "run_balanced": True,
        "focal_projection_columns": 10,
        "base_prediction_columns": 2,
    }
    registry = {
        "schema_version": "synthetic-1",
        "analysis_freeze_commit": "synthetic-freeze",
        "referee3_config_sha256": config_hash,
        "archive_name": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "fallback_specification": fallback_specification,
        "members": registry_members,
    }
    registry_path = project / "configs/referee3_heldout_registry.json"
    _write_json(registry_path, registry)
    _write_json(
        work / "tjunction/archive_inventory.json",
        {
            "archive_sha256": registry["archive_sha256"],
            "members": registry_members,
            "performance_analysis_run": False,
        },
    )

    sentinel = project / "data/processed/referee3_deployed_sentinel.npz"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_bytes(b"synthetic sentinel")
    implementation_paths = {
        "heldout_generator": project / "experiments/referee3_generate_heldout.py",
        "prediction_generator": project / "experiments/referee3_generate_predictions.py",
        "datasets": project / "src/pedgeom/datasets.py",
        "calibration": project / "src/pedgeom/calibration.py",
        "graph": project / "src/pedgeom/graph_baseline.py",
        "benchmarks": project / "src/pedgeom/benchmarks.py",
    }
    for name, path in implementation_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic {name}\n", encoding="utf-8")

    manifest_rows = []
    for index in range(7):
        path = project / f"data/processed/referee3_predictions/oof_cal-{index}.npz"
        count = _write_bundle(
            path,
            run=f"cal-{index}",
            role="calibration",
            seeds=seeds,
            heldout=False,
        )
        manifest_rows.append(
            {
                "path": str(path.relative_to(project)),
                "run": f"cal-{index}",
                "role": "calibration",
                "samples": count,
                "members": len(seeds),
                "member_seeds": seeds,
                "sha256": _sha256(path),
            }
        )
    _write_json(
        project / "data/processed/referee3_training_manifest.json",
        {"config_sha256": config_hash, "bundles": manifest_rows},
    )

    cache_directory = work / "referee3_member_cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    fits = []
    for seed in seeds:
        cache = cache_directory / f"member_{seed}.npz"
        cache.write_bytes(f"cache-{seed}".encode())
        fits.append(
            {
                "cache": "heldout_frozen_final_pool",
                "cache_file": cache.name,
                "cache_sha256": _sha256(cache),
                "cache_signature": "synthetic",
                "seed": seed,
                "wall_features": False,
                "evaluation_target_blind": True,
            }
        )

    bundle_rows = []
    for index, member in enumerate(registry_members):
        run = Path(member["name"]).stem
        path = predictions / f"heldout_{run}.npz"
        count = _write_bundle(
            path,
            run=run,
            role=analysis.EXPECTED_ROLE,
            seeds=seeds,
            heldout=True,
            shift=0.60 + 0.02 * index,
        )
        bundle_rows.append(
            {
                "file": path.name,
                "run": run,
                "samples": count,
                "members": len(seeds),
                "member_seeds": seeds,
                "sha256": _sha256(path),
            }
        )
    receipt = {
        "receipt_schema_version": 1,
        "analysis_freeze_commit": registry["analysis_freeze_commit"],
        "referee3_config_sha256": config_hash,
        "heldout_registry_sha256": _sha256(registry_path),
        "archive_sha256": registry["archive_sha256"],
        "archive_members": [row["name"] for row in registry_members],
        "deployed_sentinel_sha256": _sha256(sentinel),
        "deployed_sentinel_verified": True,
        "pool_size": len(seeds),
        "seeds": seeds,
        "fits": fits,
        "bundles": bundle_rows,
        "target_labels_used_for_prediction": False,
        "performance_analysis_run": False,
        "fallback_specification": fallback_specification,
        "implementation_sha256": {
            name: _sha256(path) for name, path in implementation_paths.items()
        },
    }
    _write_json(predictions / "ONE_SHOT_PREDICTION_RECEIPT.json", receipt)
    return project, predictions, output


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    values = np.array([0.04, 0.01, 0.03, 0.20])
    adjusted = analysis.holm_adjust(values)
    assert np.allclose(adjusted, [0.09, 0.04, 0.09, 0.20])
    order = np.argsort(values)
    assert np.all(np.diff(adjusted[order]) >= 0.0)


def test_receipt_rejects_role_change_even_with_updated_bundle_hash(tmp_path: Path) -> None:
    project, predictions, _ = _synthetic_frozen_project(tmp_path)
    receipt_path = predictions / "ONE_SHOT_PREDICTION_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    row = receipt["bundles"][0]
    path = predictions / row["file"]
    with np.load(path, allow_pickle=False) as payload:
        values = {name: np.array(payload[name]) for name in payload.files}
    values["role"] = np.asarray("familiar")
    np.savez_compressed(path, **values)
    row["sha256"] = _sha256(path)
    _write_json(receipt_path, receipt)

    with pytest.raises(RuntimeError, match="bundle role mismatch"):
        analysis.verify_prediction_receipt(project, predictions)


def test_receipt_rejects_tampered_bundle_before_analysis(tmp_path: Path) -> None:
    project, predictions, _ = _synthetic_frozen_project(tmp_path)
    bundle = next(predictions.glob("heldout_*.npz"))
    bundle.write_bytes(bundle.read_bytes() + b"tamper")

    with pytest.raises(RuntimeError, match="bundle checksum mismatch"):
        analysis.verify_prediction_receipt(project, predictions)


def test_existing_final_output_refuses_analysis_without_creating_marker(
    tmp_path: Path,
) -> None:
    project, predictions, output = _synthetic_frozen_project(tmp_path)
    output.mkdir(parents=True, exist_ok=True)
    existing = output / "referee3_heldout_summary.md"
    existing.write_text("preserve me", encoding="utf-8")

    with pytest.raises(RuntimeError, match="output already exists"):
        analysis.analyse_heldout(predictions, output, project=project)

    assert existing.read_text(encoding="utf-8") == "preserve me"
    assert not (predictions / "ONE_SHOT_ANALYSIS_RECEIPT.json").exists()


def test_post_reservation_failure_is_persistent_and_forbids_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project, predictions, output = _synthetic_frozen_project(tmp_path)

    def fail_after_reservation(*_args, **_kwargs):
        raise ArithmeticError("synthetic analysis failure")

    monkeypatch.setattr(analysis, "_primary_run_metrics", fail_after_reservation)
    with pytest.raises(ArithmeticError, match="synthetic analysis failure"):
        analysis.analyse_heldout(predictions, output, project=project)

    marker = predictions / "ONE_SHOT_ANALYSIS_RECEIPT.json"
    record = json.loads(marker.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error_type"] == "ArithmeticError"
    assert record["performance_analysis_run"] is False
    assert record["outputs_present_after_failure"] == {}
    with pytest.raises(RuntimeError, match="marker already exists"):
        analysis.analyse_heldout(predictions, output, project=project)


@pytest.mark.filterwarnings("ignore:Precision loss occurred")
def test_synthetic_frozen_analysis_writes_strict_publication_contract(tmp_path: Path) -> None:
    project, predictions, output = _synthetic_frozen_project(tmp_path)

    paths = analysis.analyse_heldout(predictions, output, project=project)

    assert all(path.is_file() for path in paths.values())
    metrics = pd.read_csv(paths["metrics"])
    assert metrics.columns.tolist() == list(analysis.METRICS_COLUMNS)
    assert set(metrics["model"]) == {"graph", "guard"}
    assert set(metrics["role"]) == {analysis.EXPECTED_ROLE}
    assert metrics.groupby("run")["model"].nunique().eq(2).all()
    inference = pd.read_csv(paths["inference"])
    assert inference.columns.tolist() == list(analysis.INFERENCE_COLUMNS)
    assert inference["endpoint"].tolist() == list(analysis.PRIMARY_ENDPOINTS)
    assert inference["n_runs"].eq(3).all()
    assert (inference["holm_adjusted_p"] >= inference["exact_two_sided_p"]).all()
    equivalence = pd.read_csv(paths["equivalence"])
    assert set(equivalence["margin_type"]) == {
        "absolute_mps",
        "relative_to_run_graph_rmse",
    }
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["small_run_inference"] is True
    assert payload["calibration"]["heldout_labels_used_for_calibration"] is False
    assert "few runs" in paths["summary"].read_text(encoding="utf-8")
    marker = json.loads(paths["analysis_receipt"].read_text(encoding="utf-8"))
    assert marker["status"] == "complete"
    assert marker["performance_analysis_run"] is True
    assert set(marker["outputs"]) == {
        "metrics",
        "inference",
        "equivalence",
        "json",
        "summary",
    }
    for name, evidence in marker["outputs"].items():
        assert evidence["sha256"] == _sha256(paths[name])

    before = {name: path.read_bytes() for name, path in paths.items()}
    with pytest.raises(RuntimeError, match="marker already exists"):
        analysis.analyse_heldout(predictions, output, project=project)
    assert before == {name: path.read_bytes() for name, path in paths.items()}
