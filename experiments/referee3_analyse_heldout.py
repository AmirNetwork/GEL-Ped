# Author: Amir Ghorbani
"""Analyse the frozen Juelich T-junction prediction bundles exactly once.

This module is deliberately narrower than :mod:`referee3_analyse`.  It accepts
only the primary five-member graph ensemble and its already frozen
variance-triggered guard.  Before targets are used, it verifies the prediction
receipt, the locked configuration and registry, the original archive and
inventory, every fit cache, every declared implementation hash, and every
prediction bundle.  No held-out parameter selection is available.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pedgeom.runtime_guard import one_sided_gate_weights
from pedgeom.statistics import exact_paired_randomization_pvalue
from referee3_analyse import (
    Bundle,
    _prediction_metrics,
    _stable_run_seed,
    _tost_row,
    calibrate_draw,
    load_bundle,
    selected_predictions,
    signals,
)


DEFAULT_PREDICTION_DIRECTORY = Path(
    r"E:\GEL-Ped-TRC-Review-Compute\referee3_tjunction_predictions"
)
EXPECTED_ROLE = "untouched T-junction"
EXPECTED_FIT_ROLE = "all calibration runs; frozen 25-member final pool"
PRIMARY_ENDPOINTS = ("rmse", "q95", "cvar95", "rate_error_gt_0_5")
METRICS_COLUMNS = (
    "run",
    "role",
    "model",
    "n",
    *PRIMARY_ENDPOINTS,
)
INFERENCE_COLUMNS = (
    "endpoint",
    "n_runs",
    "graph_run_macro",
    "guard_run_macro",
    "mean_guard_minus_graph",
    "median_guard_minus_graph",
    "relative_mean_difference_percent",
    "wins_guard",
    "ties",
    "ci95_lower",
    "ci95_upper",
    "exact_two_sided_p",
    "holm_adjusted_p",
    "reject_holm_0_05",
    "inference_unit",
    "small_run_caution",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read valid JSON from {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exclusive_write_text(path: Path, text: str) -> None:
    """Publish a final artefact without ever replacing an existing file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise RuntimeError(f"refusing to overwrite one-shot output: {path}") from error


def _atomic_replace_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically update the analysis marker owned by this reservation."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _one_shot_paths(
    prediction_directory: Path,
    output_directory: Path,
) -> tuple[Path, dict[str, Path]]:
    marker = prediction_directory / "ONE_SHOT_ANALYSIS_RECEIPT.json"
    outputs = {
        "metrics": output_directory / "referee3_heldout_primary_run_metrics.csv",
        "inference": output_directory / "referee3_heldout_confirmatory_inference.csv",
        "equivalence": output_directory / "referee3_heldout_equivalence.csv",
        "json": output_directory / "referee3_heldout_analysis.json",
        "summary": output_directory / "referee3_heldout_summary.md",
    }
    return marker, outputs


def _refuse_existing_one_shot_files(marker: Path, outputs: dict[str, Path]) -> None:
    if marker.exists():
        raise RuntimeError(
            "one-shot held-out analysis marker already exists; refusing to analyse again: "
            f"{marker}"
        )
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise RuntimeError(
            "one-shot held-out output already exists; refusing to overwrite: "
            + ", ".join(existing)
        )


def _output_evidence(outputs: dict[str, Path]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for name, path in outputs.items():
        if path.is_file():
            evidence[name] = {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return evidence


def _expected_member_seeds(config: dict[str, Any]) -> list[int]:
    replication = config["replication"]
    base = int(replication["member_seed_base"])
    stride = int(replication["member_seed_stride"])
    count = int(replication["member_pool_size"])
    return [base + stride * index for index in range(count)]


def _require_sha256(value: object, context: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise RuntimeError(f"invalid SHA256 in {context}")
    return text


def _read_bundle_contract(
    path: Path,
    *,
    expected_run: str,
    expected_samples: int,
    expected_seeds: list[int],
) -> Bundle:
    """Load one bundle only after its frozen metadata contract is checked."""

    try:
        with np.load(path, allow_pickle=False) as payload:
            schema = int(payload["bundle_schema_version"].item())
            run = str(payload["run"].item())
            role = str(payload["role"].item())
            fit_role = str(payload["fit_role"].item())
            member_seeds = np.asarray(payload["member_seeds"], dtype=np.int64)
            target = np.asarray(payload["target"], dtype=float)
            members = np.asarray(payload["members_raw"], dtype=float)
    except (KeyError, OSError, ValueError) as error:
        raise RuntimeError(f"invalid held-out prediction bundle: {path}") from error
    if schema != 1:
        raise RuntimeError(f"unsupported held-out bundle schema in {path.name}")
    if run != expected_run:
        raise RuntimeError(f"held-out bundle run mismatch in {path.name}")
    if role != EXPECTED_ROLE:
        raise RuntimeError(f"held-out bundle role mismatch in {path.name}")
    if fit_role != EXPECTED_FIT_ROLE:
        raise RuntimeError(f"held-out bundle fit role mismatch in {path.name}")
    if not np.array_equal(member_seeds, np.asarray(expected_seeds, dtype=np.int64)):
        raise RuntimeError(f"held-out bundle member seeds mismatch in {path.name}")
    if target.shape != (expected_samples, 2):
        raise RuntimeError(f"held-out bundle sample count mismatch in {path.name}")
    if members.shape != (len(expected_seeds), expected_samples, 2):
        raise RuntimeError(f"held-out bundle member shape mismatch in {path.name}")
    bundle = load_bundle(path)
    if bundle.run != expected_run or bundle.role != EXPECTED_ROLE:
        raise RuntimeError(f"held-out bundle metadata mismatch in {path.name}")
    return bundle


def _verify_inventory(
    prediction_directory: Path,
    registry: dict[str, Any],
) -> dict[str, Any]:
    work_directory = prediction_directory.parent
    archive_path = work_directory / "tjunction" / str(registry["archive_name"])
    inventory_path = work_directory / "tjunction" / "archive_inventory.json"
    if not archive_path.is_file():
        raise RuntimeError(f"registered held-out archive is missing: {archive_path}")
    if _sha256(archive_path) != str(registry["archive_sha256"]):
        raise RuntimeError("registered held-out archive checksum mismatch")
    if archive_path.stat().st_size != int(registry["archive_bytes"]):
        raise RuntimeError("registered held-out archive byte count mismatch")
    inventory = _load_json(inventory_path)
    if inventory.get("archive_sha256") != registry["archive_sha256"]:
        raise RuntimeError("held-out inventory archive checksum mismatch")
    if inventory.get("members") != registry["members"]:
        raise RuntimeError("held-out inventory member list differs from registry")
    if inventory.get("performance_analysis_run") is not False:
        raise RuntimeError("held-out inventory no longer records a pre-analysis state")
    return {
        "archive_path": str(archive_path),
        "archive_sha256": str(registry["archive_sha256"]),
        "inventory_path": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
    }


def verify_prediction_receipt(
    project: Path,
    prediction_directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[Bundle], dict[str, Any]]:
    """Verify the frozen chain of custody before exposing any target values."""

    project = project.resolve()
    prediction_directory = prediction_directory.resolve()
    config_path = project / "configs/referee3_lock.json"
    registry_path = project / "configs/referee3_heldout_registry.json"
    receipt_path = prediction_directory / "ONE_SHOT_PREDICTION_RECEIPT.json"
    config = _load_json(config_path)
    registry = _load_json(registry_path)
    receipt = _load_json(receipt_path)

    config_sha256 = _sha256(config_path)
    registry_sha256 = _sha256(registry_path)
    if str(registry.get("referee3_config_sha256")) != config_sha256:
        raise RuntimeError("held-out registry does not match the frozen configuration")
    if receipt.get("receipt_schema_version") != 1:
        raise RuntimeError("unsupported one-shot prediction receipt schema")
    if receipt.get("analysis_freeze_commit") != registry.get("analysis_freeze_commit"):
        raise RuntimeError("receipt and registry freeze commits differ")
    if receipt.get("referee3_config_sha256") != config_sha256:
        raise RuntimeError("receipt configuration checksum mismatch")
    if receipt.get("heldout_registry_sha256") != registry_sha256:
        raise RuntimeError("receipt registry checksum mismatch")
    if receipt.get("archive_sha256") != registry.get("archive_sha256"):
        raise RuntimeError("receipt archive checksum mismatch")
    expected_archive_members = [str(row["name"]) for row in registry["members"]]
    if receipt.get("archive_members") != expected_archive_members:
        raise RuntimeError("receipt archive members differ from the registry")
    if receipt.get("deployed_sentinel_verified") is not True:
        raise RuntimeError("receipt does not confirm deployed-sentinel verification")
    if receipt.get("target_labels_used_for_prediction") is not False:
        raise RuntimeError("receipt does not confirm target-blind prediction")
    if receipt.get("performance_analysis_run") is not False:
        raise RuntimeError("receipt indicates that performance was already analysed")
    if receipt.get("fallback_specification") != registry.get("fallback_specification"):
        raise RuntimeError("receipt fallback specification differs from the registry")

    expected_seeds = _expected_member_seeds(config)
    if receipt.get("pool_size") != len(expected_seeds) or receipt.get("seeds") != expected_seeds:
        raise RuntimeError("receipt member pool differs from the frozen configuration")
    primary_indices = [int(value) for value in config["replication"]["primary_member_indices"]]
    if primary_indices != list(range(len(primary_indices))):
        raise RuntimeError("held-out implementation requires the frozen leading primary members")
    if len(primary_indices) != int(config["replication"]["primary_ensemble_size"]):
        raise RuntimeError("primary member indices and ensemble size are inconsistent")

    sentinel_path = project / "data/processed/referee3_deployed_sentinel.npz"
    if not sentinel_path.is_file():
        raise RuntimeError("frozen deployed sentinel is missing")
    if _sha256(sentinel_path) != receipt.get("deployed_sentinel_sha256"):
        raise RuntimeError("deployed sentinel checksum mismatch")

    implementation_paths = {
        "heldout_generator": project / "experiments/referee3_generate_heldout.py",
        "prediction_generator": project / "experiments/referee3_generate_predictions.py",
        "datasets": project / "src/pedgeom/datasets.py",
        "calibration": project / "src/pedgeom/calibration.py",
        "graph": project / "src/pedgeom/graph_baseline.py",
        "benchmarks": project / "src/pedgeom/benchmarks.py",
    }
    declared_implementations = receipt.get("implementation_sha256")
    if not isinstance(declared_implementations, dict):
        raise RuntimeError("receipt implementation hashes are missing")
    if set(declared_implementations) != set(implementation_paths):
        raise RuntimeError("receipt implementation hash set is incomplete")
    for name, path in implementation_paths.items():
        if not path.is_file() or _sha256(path) != declared_implementations[name]:
            raise RuntimeError(f"receipt implementation checksum mismatch: {name}")

    fits = receipt.get("fits")
    if not isinstance(fits, list) or len(fits) != len(expected_seeds):
        raise RuntimeError("receipt does not declare every frozen member fit")
    cache_directory = prediction_directory.parent / "referee3_member_cache"
    fit_seeds: list[int] = []
    for row in fits:
        if not isinstance(row, dict):
            raise RuntimeError("invalid fit record in prediction receipt")
        seed = int(row.get("seed", -1))
        fit_seeds.append(seed)
        cache_name = str(row.get("cache_file", ""))
        if Path(cache_name).name != cache_name:
            raise RuntimeError("fit-cache receipt contains an unsafe path")
        cache_path = cache_directory / cache_name
        declared_hash = _require_sha256(row.get("cache_sha256"), "fit-cache receipt")
        if not cache_path.is_file() or _sha256(cache_path) != declared_hash:
            raise RuntimeError(f"fit-cache checksum mismatch: {cache_name}")
        if row.get("evaluation_target_blind") is not True:
            raise RuntimeError(f"fit-cache was not target-blind: {cache_name}")
        if row.get("wall_features") is not False:
            raise RuntimeError(f"fit-cache used forbidden wall features: {cache_name}")
    if sorted(fit_seeds) != expected_seeds or len(set(fit_seeds)) != len(fit_seeds):
        raise RuntimeError("fit-cache seeds differ from the frozen member pool")

    declared_bundles = receipt.get("bundles")
    if not isinstance(declared_bundles, list):
        raise RuntimeError("receipt bundle declaration is missing")
    expected_runs = [Path(name).stem for name in expected_archive_members]
    if [row.get("run") for row in declared_bundles] != expected_runs:
        raise RuntimeError("receipt run order differs from the held-out registry")
    declared_files: set[str] = set()
    bundles: list[Bundle] = []
    for row in declared_bundles:
        if not isinstance(row, dict):
            raise RuntimeError("invalid bundle record in prediction receipt")
        file_name = str(row.get("file", ""))
        if Path(file_name).name != file_name or not file_name.endswith(".npz"):
            raise RuntimeError("prediction receipt contains an unsafe bundle path")
        if file_name in declared_files:
            raise RuntimeError(f"duplicate bundle in prediction receipt: {file_name}")
        declared_files.add(file_name)
        path = prediction_directory / file_name
        declared_hash = _require_sha256(row.get("sha256"), "bundle receipt")
        if not path.is_file() or _sha256(path) != declared_hash:
            raise RuntimeError(f"prediction bundle checksum mismatch: {file_name}")
        bundles.append(
            _read_bundle_contract(
                path,
                expected_run=str(row["run"]),
                expected_samples=int(row["samples"]),
                expected_seeds=expected_seeds,
            )
        )
        if int(row.get("members", -1)) != len(expected_seeds):
            raise RuntimeError(f"receipt member count mismatch: {file_name}")
        if row.get("member_seeds") != expected_seeds:
            raise RuntimeError(f"receipt member seeds mismatch: {file_name}")
    actual_files = {path.name for path in prediction_directory.glob("*.npz")}
    if actual_files != declared_files:
        raise RuntimeError("prediction directory has undeclared or missing bundles")
    minimum_runs = int(config["heldout"]["minimum_complete_runs"])
    if len(bundles) < minimum_runs:
        raise RuntimeError("too few complete held-out runs for frozen inference")

    inventory_verification = _verify_inventory(prediction_directory, registry)
    verification = {
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256(receipt_path),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "registry_path": str(registry_path),
        "registry_sha256": registry_sha256,
        "sentinel_path": str(sentinel_path),
        "sentinel_sha256": _sha256(sentinel_path),
        "verified_bundle_count": len(bundles),
        "verified_fit_cache_count": len(fits),
        "expected_role": EXPECTED_ROLE,
        **inventory_verification,
    }
    return config, registry, bundles, verification


def _verified_oof_bundles(project: Path, config_sha256: str) -> list[Bundle]:
    """Load only checksum-verified calibration folds, never familiar test bundles."""

    manifest_path = project / "data/processed/referee3_training_manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("config_sha256") != config_sha256:
        raise RuntimeError("training manifest does not match the frozen configuration")
    rows = manifest.get("bundles")
    if not isinstance(rows, list):
        raise RuntimeError("training manifest bundle list is missing")
    logical_bundle_root = Path("data/processed/referee3_predictions")
    resolved_bundle_root = (project / logical_bundle_root).resolve()
    oof: list[Bundle] = []
    seen_runs: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("invalid training-manifest bundle record")
        relative = Path(str(row.get("path", "")))
        try:
            relative.relative_to(logical_bundle_root)
        except ValueError as error:
            raise RuntimeError(
                "OOF calibration bundle is outside the frozen prediction directory"
            ) from error
        path = (project / relative).resolve()
        if not path.is_relative_to(resolved_bundle_root):
            raise RuntimeError("training-manifest bundle path leaves its declared directory")
        if not path.name.startswith("oof_"):
            continue
        if not path.is_file() or _sha256(path) != row.get("sha256"):
            raise RuntimeError(f"OOF calibration bundle checksum mismatch: {path.name}")
        if row.get("role") != "calibration":
            raise RuntimeError(f"OOF calibration role mismatch: {path.name}")
        try:
            with np.load(path, allow_pickle=False) as payload:
                member_seeds = np.asarray(payload["member_seeds"], dtype=np.int64)
                fit_role = str(payload["fit_role"].item())
                sample_count = len(payload["target"])
                member_count = int(payload["members_raw"].shape[0])
        except (KeyError, OSError, ValueError) as error:
            raise RuntimeError(f"invalid OOF calibration bundle: {path.name}") from error
        if fit_role != "leave-one-complete-run-out":
            raise RuntimeError(f"OOF calibration fit role mismatch: {path.name}")
        if member_seeds.tolist() != row.get("member_seeds"):
            raise RuntimeError(f"OOF calibration member seeds mismatch: {path.name}")
        if sample_count != int(row.get("samples", -1)):
            raise RuntimeError(f"OOF calibration sample count mismatch: {path.name}")
        if member_count != int(row.get("members", -1)):
            raise RuntimeError(f"OOF calibration member count mismatch: {path.name}")
        bundle = load_bundle(path)
        if bundle.run != str(row.get("run")) or bundle.role != str(row.get("role")):
            raise RuntimeError(f"OOF calibration bundle metadata mismatch: {path.name}")
        if bundle.run in seen_runs:
            raise RuntimeError(f"duplicate OOF calibration run: {bundle.run}")
        seen_runs.add(bundle.run)
        oof.append(bundle)
    if len(oof) != 7:
        raise RuntimeError(f"expected 7 frozen OOF calibration runs, found {len(oof)}")
    return oof


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return monotone Holm-adjusted p-values in their original order."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("p_values must be a non-empty vector")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must be finite and lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    scaled = (len(values) - np.arange(len(values))) * values[order]
    monotone = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(monotone)
    adjusted[order] = monotone
    return adjusted


def _run_bootstrap_ci(
    differences: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("paired run differences must be a finite vector of length >= 2")
    if repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(repetitions, len(values)))
    means = np.mean(values[indices], axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _primary_run_metrics(
    bundles: list[Bundle],
    oof: list[Bundle],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    indices = tuple(int(value) for value in config["replication"]["primary_member_indices"])
    calibration = calibrate_draw(oof, indices, config)
    cap = float(config["experts"]["fallback_prediction_speed_cap_mps"])
    window = float(config["guard"]["trailing_window_seconds"])
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        members, graph, fallback = selected_predictions(bundle, indices, cap)
        spread = signals(bundle, members, graph, fallback, window)["spread"]
        graph_weight = one_sided_gate_weights(
            spread,
            threshold=calibration.tau,
            scale=calibration.scale,
            kind=str(config["guard"]["primary_shape"]),
        )
        guard = graph_weight[:, None] * graph + (1.0 - graph_weight[:, None]) * fallback
        for model, prediction in (("graph", graph), ("guard", guard)):
            metrics = _prediction_metrics(bundle.target, prediction)
            rows.append(
                {
                    "run": bundle.run,
                    "role": bundle.role,
                    "model": model,
                    "n": len(bundle.target),
                    "rmse": metrics["rmse"],
                    "q95": metrics["q95"],
                    "cvar95": metrics["cvar95"],
                    "rate_error_gt_0_5": metrics["large_error_rate"],
                }
            )
    frame = pd.DataFrame(rows, columns=METRICS_COLUMNS)
    calibration_payload = {
        "member_indices": list(calibration.indices),
        "tau": calibration.tau,
        "scale": calibration.scale,
        "guard_shape": str(config["guard"]["primary_shape"]),
        "trailing_window_seconds": window,
        "fallback_speed_cap_mps": cap,
        "calibration_data": "seven checksum-verified OOF Juelich calibration runs",
        "heldout_labels_used_for_calibration": False,
    }
    return frame, calibration_payload


def _paired_inference(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    repetitions = int(config["endpoints"]["bootstrap_replicates"])
    base_seed = int(config["endpoints"]["bootstrap_seed"])
    rows: list[dict[str, Any]] = []
    for endpoint in PRIMARY_ENDPOINTS:
        pivot = metrics.pivot(index="run", columns="model", values=endpoint).sort_index()
        if list(pivot.columns) != ["graph", "guard"] or pivot.isna().any().any():
            raise RuntimeError(f"incomplete paired held-out endpoint: {endpoint}")
        graph = pivot["graph"].to_numpy(float)
        guard = pivot["guard"].to_numpy(float)
        difference = guard - graph
        relative = np.divide(
            difference,
            graph,
            out=np.full_like(difference, np.nan),
            where=np.abs(graph) > 1e-12,
        )
        finite_relative = relative[np.isfinite(relative)]
        relative_effect = (
            float(100.0 * np.mean(finite_relative)) if len(finite_relative) else None
        )
        low, high = _run_bootstrap_ci(
            difference,
            repetitions=repetitions,
            seed=_stable_run_seed(base_seed, f"heldout|{endpoint}"),
        )
        rows.append(
            {
                "endpoint": endpoint,
                "n_runs": len(difference),
                "graph_run_macro": float(np.mean(graph)),
                "guard_run_macro": float(np.mean(guard)),
                "mean_guard_minus_graph": float(np.mean(difference)),
                "median_guard_minus_graph": float(np.median(difference)),
                "relative_mean_difference_percent": relative_effect,
                "wins_guard": int(np.sum(difference < -1e-15)),
                "ties": int(np.sum(np.abs(difference) <= 1e-15)),
                "ci95_lower": low,
                "ci95_upper": high,
                "exact_two_sided_p": exact_paired_randomization_pvalue(difference),
                "inference_unit": "complete T-junction run",
                "small_run_caution": (
                    "Only complete-run replication is treated as independent; "
                    "with few runs, p-values are discrete and intervals are imprecise."
                ),
            }
        )
    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p"] = holm_adjust(frame["exact_two_sided_p"].to_numpy(float))
    frame["reject_holm_0_05"] = frame["holm_adjusted_p"] < 0.05
    return frame.loc[:, INFERENCE_COLUMNS]


def _equivalence(metrics: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pivot = metrics.pivot(index="run", columns="model", values="rmse").sort_index()
    absolute = (pivot["guard"] - pivot["graph"]).to_numpy(float)
    graph = pivot["graph"].to_numpy(float)
    relative = np.divide(
        absolute,
        graph,
        out=np.full_like(absolute, np.nan),
        where=np.abs(graph) > 1e-12,
    )
    absolute_margin = float(config["endpoints"]["equivalence_absolute_margin_mps"])
    relative_margin = float(config["endpoints"]["equivalence_relative_margin"])
    rows = [
        {
            "endpoint": "rmse",
            "margin_type": "absolute_mps",
            **_tost_row(absolute, absolute_margin),
        },
        {
            "endpoint": "rmse",
            "margin_type": "relative_to_run_graph_rmse",
            **_tost_row(relative, relative_margin),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["inference_unit"] = "complete T-junction run"
    frame["small_run_caution"] = (
        "TOST uses a paired-run t model; interpret cautiously because the number of runs is small."
    )
    return frame


def _summary_markdown(
    metrics: pd.DataFrame,
    inference: pd.DataFrame,
    equivalence: pd.DataFrame,
    verification: dict[str, Any],
) -> str:
    lines = [
        "# Frozen Juelich T-junction confirmatory analysis",
        "",
        "This file is machine-generated from checksum-verified one-shot prediction bundles. "
        "No threshold, expert, endpoint, inclusion rule, or ensemble member was selected using "
        "the T-junction labels.",
        "",
        f"Verified complete runs: **{metrics['run'].nunique()}**.  Negative differences favour "
        "the guarded predictor.",
        "",
        "## Confirmatory family",
        "",
        "| Endpoint | Graph macro | Guard macro | Guard - graph | 95% run bootstrap CI | "
        "Exact p | Holm p | Guard wins |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in inference.itertuples(index=False):
        lines.append(
            f"| {row.endpoint} | {row.graph_run_macro:.6g} | {row.guard_run_macro:.6g} | "
            f"{row.mean_guard_minus_graph:.6g} | [{row.ci95_lower:.6g}, "
            f"{row.ci95_upper:.6g}] | {row.exact_two_sided_p:.6g} | "
            f"{row.holm_adjusted_p:.6g} | {row.wins_guard}/{row.n_runs} |"
        )
    lines.extend(
        [
            "",
            "## Frozen RMSE equivalence tests",
            "",
            "| Margin | Estimate | 90% CI | TOST p | Equivalent at 0.05 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in equivalence.to_dict(orient="records"):
        lines.append(
            f"| {row['margin_type']} ({row['margin']:.6g}) | "
            f"{row['mean_difference']:.6g} | "
            f"[{row['ci90_lower']:.6g}, {row['ci90_upper']:.6g}] | "
            f"{row['tost_p']:.6g} | "
            f"{'yes' if row['equivalent_at_0.05'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The independent unit is a complete T-junction run. With few runs, exact sign-flip "
            "p-values are discrete, bootstrap intervals are imprecise, "
            "and a non-significant result is not evidence of no effect. The two one-sided "
            "equivalence tests are reported separately from the Holm-adjusted superiority family.",
            "",
            "## Verification",
            "",
            f"- Receipt SHA256: `{verification['receipt_sha256']}`",
            f"- Frozen configuration SHA256: `{verification['config_sha256']}`",
            f"- Held-out registry SHA256: `{verification['registry_sha256']}`",
            f"- Original archive SHA256: `{verification['archive_sha256']}`",
            f"- Verified prediction bundles: {verification['verified_bundle_count']}",
            f"- Verified member-fit caches: {verification['verified_fit_cache_count']}",
            "",
        ]
    )
    return "\n".join(lines)


def analyse_heldout(
    prediction_directory: Path,
    output_directory: Path,
    *,
    project: Path | None = None,
) -> dict[str, Path]:
    """Validate, reserve once, analyse, and publish without overwriting."""

    project = project_root() if project is None else project.resolve()
    prediction_directory = prediction_directory.resolve()
    output_directory = output_directory.resolve()
    marker, paths = _one_shot_paths(prediction_directory, output_directory)
    # Fast preflight prevents a completed or interrupted run from touching the
    # labels again.  The same check is repeated after potentially slow input
    # verification to close the ordinary race window.
    _refuse_existing_one_shot_files(marker, paths)
    config, registry, bundles, verification = verify_prediction_receipt(
        project, prediction_directory
    )
    oof = _verified_oof_bundles(project, verification["config_sha256"])
    training_manifest = project / "data/processed/referee3_training_manifest.json"
    verification["training_manifest_path"] = str(training_manifest)
    verification["training_manifest_sha256"] = _sha256(training_manifest)
    verification["verified_oof_bundle_count"] = len(oof)
    expected_family = list(config["multiplicity"]["confirmatory_family"])
    frozen_family = [
        "heldout_graph_vs_guard_run_rmse",
        "heldout_graph_vs_guard_q95",
        "heldout_graph_vs_guard_cvar95",
        "heldout_graph_vs_guard_error_gt_0.5",
    ]
    if expected_family != frozen_family or config["multiplicity"]["adjustment"] != "Holm":
        raise RuntimeError("configuration no longer matches the frozen confirmatory family")
    _refuse_existing_one_shot_files(marker, paths)
    running_marker = {
        "analysis_marker_schema_version": 1,
        "status": "running",
        "reserved_at_utc": _utc_now(),
        "performance_analysis_run": False,
        "prediction_directory": str(prediction_directory),
        "input_verification": verification,
        "intended_outputs": {name: str(path) for name, path in paths.items()},
        "rerun_policy": (
            "This marker is an irreversible reservation. A running, failed, or complete "
            "marker forbids another held-out analysis."
        ),
    }
    _exclusive_write_text(marker, json.dumps(running_marker, indent=2) + "\n")

    try:
        metrics, calibration = _primary_run_metrics(bundles, oof, config)
        inference = _paired_inference(metrics, config)
        equivalence = _equivalence(metrics, config)
        payload = {
            "analysis_schema_version": 1,
            "analysis_mode": "frozen one-shot held-out confirmation",
            "dataset": config["heldout"]["dataset"],
            "role": EXPECTED_ROLE,
            "verification": verification,
            "calibration": calibration,
            "confirmatory_endpoints": list(PRIMARY_ENDPOINTS),
            "multiplicity_adjustment": "Holm",
            "bootstrap": {
                "unit": "complete T-junction run",
                "repetitions": int(config["endpoints"]["bootstrap_replicates"]),
                "seed": int(config["endpoints"]["bootstrap_seed"]),
            },
            "run_metrics": metrics.to_dict(orient="records"),
            "confirmatory_inference": inference.to_dict(orient="records"),
            "equivalence": equivalence.to_dict(orient="records"),
            "small_run_inference": True,
            "interpretation_boundary": (
                "Complete runs are the independent units. Exact p-values are discrete, "
                "bootstrap intervals are imprecise, and no universal shift claim is made."
            ),
        }
        contents = {
            "metrics": metrics.to_csv(index=False),
            "inference": inference.to_csv(index=False),
            "equivalence": equivalence.to_csv(index=False),
            "json": json.dumps(payload, indent=2) + "\n",
            "summary": _summary_markdown(
                metrics, inference, equivalence, verification
            ),
        }
        for name in ("metrics", "inference", "equivalence", "json", "summary"):
            _exclusive_write_text(paths[name], contents[name])
        complete_marker = {
            **running_marker,
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "performance_analysis_run": True,
            "outputs": _output_evidence(paths),
        }
        _atomic_replace_json(marker, complete_marker)
    except BaseException as error:
        failed_marker = {
            **running_marker,
            "status": "failed",
            "failed_at_utc": _utc_now(),
            "performance_analysis_run": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "outputs_present_after_failure": _output_evidence(paths),
            "recovery_policy": (
                "Preserve this marker and all partial outputs for audit; do not retry or "
                "overwrite after held-out labels were exposed."
            ),
        }
        try:
            _atomic_replace_json(marker, failed_marker)
        except BaseException:
            # The original running marker remains as irreversible failure
            # evidence even if finalising the diagnostic record itself fails.
            pass
        raise
    return {**paths, "analysis_receipt": marker}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen one-shot Juelich T-junction analysis."
    )
    parser.add_argument(
        "--prediction-directory",
        type=Path,
        default=DEFAULT_PREDICTION_DIRECTORY,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=project_root() / "reports/generated",
    )
    args = parser.parse_args()
    paths = analyse_heldout(args.prediction_directory, args.output_directory)
    print(f"wrote frozen held-out analysis to {paths['summary']}", flush=True)


if __name__ == "__main__":
    main()
