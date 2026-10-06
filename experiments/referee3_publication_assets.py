# Author: Amir Ghorbani
"""Build referee-3 publication tables and vector figures from frozen outputs.

This module is deliberately presentation-only.  It never reads prediction
bundles, refits a model, changes an analysis choice, or fills absent results.
Each input is validated against the producing experiment's public schema, and
paired comparisons are accepted only when their run keys match exactly.

The generated ``.tex`` files contain their own captions and labels so that a
manuscript can include them with ``\\input`` and refer to them with the
hyperlinked ``\\tabref`` helper.  The figure-snippet file does the analogous
job for the vector PDF figures.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


class AssetSchemaError(ValueError):
    """Raised when an analysis output cannot support the declared asset."""


PRIMARY_MODELS = (
    "graph ensemble",
    "GEL-Ped",
    "feature-based HGB fallback",
    "calibration-fixed blend",
    "learned matched gate",
    "delayed-error online gate",
)
MATCHED_CONTROL_MODELS = (
    "GEL-Ped",
    "calibration-fixed blend",
    "learned matched gate",
    "delayed-error online gate",
)
PLANNER_POLICIES = (
    "direct forecast, fixed margin",
    "guarded forecast, fixed margin",
    "guarded forecast, g-adaptive margin",
)
HELDOUT_ENDPOINTS = ("rmse", "q95", "cvar95", "rate_error_gt_0_5")
STRATUM_ORDER = ("inside", "0--1 IQR", "1--2 IQR", ">2 IQR")

MODEL_LABELS = {
    "graph ensemble": "Graph ensemble",
    "GEL-Ped": "GEL-Ped",
    "feature-based HGB fallback": "HGB fallback",
    "calibration-fixed blend": "Fixed blend",
    "learned matched gate": "Learned matched gate",
    "delayed-error online gate": "Delayed-error gate",
    "transformer ensemble": "Transformer ensemble",
    "guarded transformer": "Guarded Transformer",
    "graph": "Graph ensemble",
    "guard": "GEL-Ped",
}

COLORS = {
    "navy": "#174A6E",
    "orange": "#D86B27",
    "green": "#2A8C73",
    "purple": "#7768AE",
    "red": "#B94345",
    "grey": "#6D7278",
    "light_grey": "#D9DDE1",
}
SERIES_COLORS = (
    COLORS["navy"],
    COLORS["orange"],
    COLORS["green"],
    COLORS["purple"],
    COLORS["red"],
)


@dataclass(frozen=True)
class AssetInputs:
    draw_inference: Path
    primary_metrics: Path
    targeting: Path
    decomposition: Path
    permuted_gate: Path
    heldout_metrics: Path
    heldout_inference: Path
    planner_metrics: Path
    transformer_summary: Path | None = None
    planner_latency: Path | None = None
    training_manifest: Path | None = None
    transformer_manifest: Path | None = None
    config: Path | None = None


@dataclass(frozen=True)
class AssetOutputs:
    tables: tuple[Path, ...]
    figures: tuple[Path, ...]
    snippets: Path
    manifest: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _configure_matplotlib() -> None:
    """Match the manuscript's Latin Modern serif appearance."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.3,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _read_csv(
    path: Path,
    required: Sequence[str],
    description: str,
    *,
    exact_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    _require_file(path, description)
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as error:
        raise AssetSchemaError(f"could not read {description}: {error}") from error
    if frame.empty:
        raise AssetSchemaError(f"{description} is empty: {path}")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AssetSchemaError(
            f"{description} is missing required columns {missing}; "
            f"found {list(frame.columns)}"
        )
    if exact_columns is not None and list(frame.columns) != list(exact_columns):
        raise AssetSchemaError(
            f"{description} columns/order differ from the frozen contract; "
            f"expected {list(exact_columns)}, found {list(frame.columns)}"
        )
    return frame


def _read_json(path: Path, description: str) -> dict[str, Any]:
    _require_file(path, description)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssetSchemaError(f"could not read {description}: {error}") from error
    if not isinstance(payload, dict):
        raise AssetSchemaError(f"{description} must contain a JSON object")
    return payload


def _require_numeric(
    frame: pd.DataFrame,
    columns: Sequence[str],
    description: str,
    *,
    allow_nan: Iterable[str] = (),
) -> None:
    nan_allowed = set(allow_nan)
    for column in columns:
        source_missing = frame[column].isna()
        converted = pd.to_numeric(frame[column], errors="coerce")
        if column in nan_allowed:
            invalid = (~source_missing & converted.isna()) | (
                converted.notna() & ~np.isfinite(converted)
            )
        else:
            invalid = converted.isna() | ~np.isfinite(converted)
        if bool(invalid.any()):
            rows = frame.index[invalid].tolist()[:5]
            raise AssetSchemaError(
                f"{description}.{column} contains non-finite/non-numeric values "
                f"at rows {rows}"
            )


def _require_unique(frame: pd.DataFrame, keys: Sequence[str], description: str) -> None:
    duplicated = frame.duplicated(list(keys), keep=False)
    if bool(duplicated.any()):
        example = frame.loc[duplicated, list(keys)].head(5).to_dict("records")
        raise AssetSchemaError(
            f"{description} has duplicate rows for keys {list(keys)}: {example}"
        )


def _require_values(
    frame: pd.DataFrame,
    column: str,
    required: Sequence[str],
    description: str,
    *,
    exact: bool = False,
) -> None:
    observed = set(frame[column].dropna().astype(str))
    expected = set(required)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected) if exact else []
    if missing or extra:
        raise AssetSchemaError(
            f"{description}.{column} violates the frozen values; "
            f"missing={missing}, extra={extra}"
        )


def _validate_complete_cross(
    frame: pd.DataFrame,
    group_keys: Sequence[str],
    value_column: str,
    required_values: Sequence[str],
    description: str,
) -> None:
    expected = set(required_values)
    failures: list[dict[str, Any]] = []
    grouper: str | list[str] = (
        group_keys[0] if len(group_keys) == 1 else list(group_keys)
    )
    for key, group in frame.groupby(grouper, sort=False, dropna=False):
        observed = set(group[value_column].astype(str))
        if observed != expected:
            key_tuple = key if isinstance(key, tuple) else (key,)
            failures.append(
                {
                    **dict(zip(group_keys, key_tuple, strict=True)),
                    "missing": sorted(expected - observed),
                    "extra": sorted(observed - expected),
                }
            )
    if failures:
        raise AssetSchemaError(
            f"{description} does not contain a complete {value_column} cross: "
            f"{failures[:5]}"
        )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _save_figure(figure: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.pdf")
    try:
        figure.savefig(
            temporary,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.025,
            metadata={
                "Title": path.stem.replace("_", " "),
                "Author": "Amir Ghorbani",
                "Creator": "GEL-Ped referee-3 publication asset builder",
                "CreationDate": datetime(2026, 10, 2, tzinfo=timezone.utc),
                "ModDate": datetime(2026, 10, 2, tzinfo=timezone.utc),
            },
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)


def _tex(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _format_number(value: float, digits: int = 4) -> str:
    if not math.isfinite(float(value)):
        return "--"
    return f"{float(value):.{digits}f}"


def _format_p(value: float) -> str:
    if not math.isfinite(float(value)):
        return "--"
    if value < 0.001:
        return r"$<0.001$"
    return f"{float(value):.3f}"


def _table(
    *,
    caption: str,
    label: str,
    column_specification: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    notes: str | None = None,
    size_command: str = r"\small",
    tabcolsep: str = "4.2pt",
) -> str:
    body = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        size_command,
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\begin{{tabular}}{{{column_specification}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.extend([r"\bottomrule", r"\end{tabular}"])
    if notes:
        body.append(rf"\par\vspace{{2pt}}\footnotesize {notes}")
    body.extend([r"\end{table}", ""])
    return "\n".join(body)


def _style_axis(axis: Axes, *, zero_line: bool = False) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#E7E9EB", linewidth=0.55, zorder=0)
    axis.tick_params(length=2.8, width=0.6)
    if zero_line:
        axis.axhline(0.0, color="#3A3A3A", linewidth=0.75, zorder=1)


def _panel_label(axis: Axes, label: str) -> None:
    axis.text(
        -0.18,
        1.04,
        label,
        transform=axis.transAxes,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _paired_model_difference(
    frame: pd.DataFrame,
    model: str,
    endpoint: str,
) -> pd.DataFrame:
    selected = frame[frame["model"].isin(("graph ensemble", model))]
    wide = selected.pivot(index=["role", "run"], columns="model", values=endpoint)
    if wide[["graph ensemble", model]].isna().any().any():
        raise AssetSchemaError(
            f"primary metrics lack a paired graph/{model} {endpoint} value"
        )
    result = wide.reset_index()
    result["difference"] = result[model] - result["graph ensemble"]
    return result


def _draw_assets(frame: pd.DataFrame, table_dir: Path, figure_dir: Path) -> tuple[Path, Path]:
    required = (
        "role",
        "ensemble_size",
        "endpoint",
        "n_runs",
        "n_draws",
        "mean_guard_minus_graph",
        "fraction_draws_with_improvement",
        "crossed_bootstrap_q025",
        "crossed_bootstrap_q975",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AssetSchemaError(f"draw inference is missing required columns {missing}")
    _require_numeric(
        frame,
        (
            "ensemble_size",
            "n_runs",
            "n_draws",
            "mean_guard_minus_graph",
            "fraction_draws_with_improvement",
            "crossed_bootstrap_q025",
            "crossed_bootstrap_q975",
        ),
        "draw inference",
    )
    _require_unique(frame, ("role", "ensemble_size", "endpoint"), "draw inference")
    _require_values(frame, "endpoint", ("rmse", "q95", "cvar95"), "draw inference")
    if bool(
        (
            frame["crossed_bootstrap_q025"]
            > frame["crossed_bootstrap_q975"]
        ).any()
    ):
        raise AssetSchemaError("draw inference contains a reversed confidence interval")
    if bool(
        ~frame["fraction_draws_with_improvement"].between(0.0, 1.0).all()
    ):
        raise AssetSchemaError("draw improvement fractions must lie in [0, 1]")
    _validate_complete_cross(
        frame[frame["endpoint"].isin(("rmse", "q95", "cvar95"))],
        ("role", "ensemble_size"),
        "endpoint",
        ("rmse", "q95", "cvar95"),
        "draw inference",
    )

    endpoints = ("rmse", "q95", "cvar95")
    rows: list[list[str]] = []
    for role in sorted(frame["role"].astype(str).unique()):
        role_frame = frame[frame["role"].eq(role)]
        for size in sorted(role_frame["ensemble_size"].astype(int).unique()):
            selection = role_frame[role_frame["ensemble_size"].eq(size)].set_index("endpoint")
            cells = []
            for endpoint in endpoints:
                current = selection.loc[endpoint]
                cells.append(
                    f"{_format_number(current['mean_guard_minus_graph'])} "
                    f"[{_format_number(current['crossed_bootstrap_q025'])}, "
                    f"{_format_number(current['crossed_bootstrap_q975'])}]"
                )
            rmse_row = selection.loc["rmse"]
            rows.append(
                [
                    _tex(role),
                    str(int(size)),
                    str(int(rmse_row["n_draws"])),
                    *cells,
                    f"{100.0 * rmse_row['fraction_draws_with_improvement']:.0f}\\%",
                ]
            )
    table_path = table_dir / "referee3_draw_stability.tex"
    _atomic_text(
        table_path,
        _table(
            caption=(
                "Member-draw stability. Entries are GEL-Ped minus graph error in "
                "m/s with crossed complete-run/member-draw 95\\% intervals."
            ),
            label="tab:r3-draw-stability",
            column_specification="llrrrrr",
            header=(
                "Regime",
                "$M$",
                "Draws",
                "$\\Delta$RMSE",
                "$\\Delta q_{.95}$",
                "$\\Delta$CVaR$_{.95}$",
                "RMSE wins",
            ),
            rows=rows,
            notes=(
                "Negative differences favour GEL-Ped. Intervals resample complete runs "
                "and complete member draws; overlapping member subsets are not treated "
                "as independent training experiments."
            ),
            size_command=r"\scriptsize",
            tabcolsep="2.4pt",
        ),
    )

    plot = frame[frame["endpoint"].isin(("rmse", "q95"))].copy()
    roles = sorted(plot["role"].astype(str).unique())
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.55), sharex=True)
    for panel, (axis, endpoint, title) in enumerate(
        zip(axes, ("rmse", "q95"), ("Mean error", "95th-percentile error"), strict=True)
    ):
        current = plot[plot["endpoint"].eq(endpoint)]
        for color, role in zip(SERIES_COLORS, roles, strict=False):
            values = current[current["role"].eq(role)].sort_values("ensemble_size")
            x = values["ensemble_size"].to_numpy(float)
            y = values["mean_guard_minus_graph"].to_numpy(float)
            low = values["crossed_bootstrap_q025"].to_numpy(float)
            high = values["crossed_bootstrap_q975"].to_numpy(float)
            axis.vlines(x, low, high, color=color, linewidth=1.05, zorder=2)
            axis.plot(x, y, marker="o", color=color, label=role, zorder=3)
            cap_width = 0.08
            axis.hlines(low, x - cap_width, x + cap_width, color=color, linewidth=1.05)
            axis.hlines(high, x - cap_width, x + cap_width, color=color, linewidth=1.05)
        _style_axis(axis, zero_line=True)
        _panel_label(axis, f"({chr(97 + panel)})")
        axis.set_title(title, pad=5)
        axis.set_xlabel("Ensemble size $M$")
        axis.set_ylabel("GEL-Ped minus graph (m/s)")
        axis.set_xticks(sorted(current["ensemble_size"].astype(int).unique()))
    axes[1].legend(frameon=False, loc="best", title="Evaluation regime")
    figure.tight_layout(w_pad=1.1)
    figure_path = figure_dir / "referee3_draw_stability.pdf"
    _save_figure(figure, figure_path)
    return table_path, figure_path


def _primary_assets(
    frame: pd.DataFrame, table_dir: Path, figure_dir: Path
) -> tuple[Path, Path]:
    required = (
        "role",
        "run",
        "model",
        "n",
        "evaluation_status",
        "rmse",
        "q95",
        "cvar95",
        "large_error_rate",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AssetSchemaError(f"primary metrics is missing required columns {missing}")
    _require_numeric(
        frame,
        ("n", "rmse", "q95", "cvar95", "large_error_rate"),
        "primary metrics",
    )
    _require_unique(frame, ("role", "run", "model"), "primary metrics")
    if bool((frame["n"] <= 0).any()):
        raise AssetSchemaError("primary metrics sample counts must be positive")
    if bool(~frame["large_error_rate"].between(0.0, 1.0).all()):
        raise AssetSchemaError("primary large-error rates must lie in [0, 1]")
    evaluation = frame[frame["evaluation_status"].eq("locked-evaluation")].copy()
    if evaluation.empty:
        raise AssetSchemaError("primary metrics contains no locked-evaluation rows")
    _require_values(evaluation, "model", PRIMARY_MODELS, "primary metrics")
    _validate_complete_cross(
        evaluation[evaluation["model"].isin(PRIMARY_MODELS)],
        ("role", "run"),
        "model",
        PRIMARY_MODELS,
        "primary metrics",
    )

    summary = (
        evaluation[evaluation["model"].isin(PRIMARY_MODELS)]
        .groupby(["role", "model"], sort=False)[
            ["rmse", "q95", "cvar95", "large_error_rate"]
        ]
        .mean()
        .reset_index()
    )
    rows: list[list[str]] = []
    for role in sorted(summary["role"].astype(str).unique()):
        role_frame = summary[summary["role"].eq(role)].set_index("model")
        for model in PRIMARY_MODELS:
            current = role_frame.loc[model]
            rows.append(
                [
                    _tex(role),
                    _tex(MODEL_LABELS[model]),
                    _format_number(current["rmse"]),
                    _format_number(current["q95"]),
                    _format_number(current["cvar95"]),
                    f"{100.0 * current['large_error_rate']:.2f}\\%",
                ]
            )
    table_path = table_dir / "referee3_tail_controls.tex"
    _atomic_text(
        table_path,
        _table(
            caption=(
                "Run-macro mean and tail error for GEL-Ped and matched controls. "
                "All methods use the same evaluation origins; gates use the same "
                "graph--fallback expert pair."
            ),
            label="tab:r3-tail-controls",
            column_specification="llrrrr",
            header=(
                "Regime",
                "Method",
                "RMSE",
                "$q_{.95}$",
                "CVaR$_{.95}$",
                "$e>0.5$",
            ),
            rows=rows,
            notes="Velocity-error quantities are in m/s. The final column is a forecast share.",
        ),
    )

    roles = sorted(evaluation["role"].astype(str).unique())
    markers = ("o", "s", "^", "D")
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), sharey=True)
    for panel, (axis, endpoint, title) in enumerate(
        zip(axes, ("q95", "cvar95"), ("95th percentile", "Upper-tail mean"), strict=True)
    ):
        positions = np.arange(len(roles), dtype=float)
        offsets = np.linspace(-0.22, 0.22, len(MATCHED_CONTROL_MODELS))
        for color, marker, offset, model in zip(
            SERIES_COLORS[: len(MATCHED_CONTROL_MODELS)],
            markers,
            offsets,
            MATCHED_CONTROL_MODELS,
            strict=True,
        ):
            differences = _paired_model_difference(evaluation, model, endpoint)
            means = differences.groupby("role", sort=False)["difference"].mean()
            axis.scatter(
                [means.loc[role] for role in roles],
                positions + offset,
                color=color,
                marker=marker,
                s=22,
                label=MODEL_LABELS[model],
                zorder=3,
            )
        axis.axvline(0.0, color="#3A3A3A", linewidth=0.75, zorder=1)
        axis.grid(axis="x", color="#E7E9EB", linewidth=0.55, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_yticks(positions, [_tex(role).replace(r"\_", "_") for role in roles])
        axis.set_xlabel("Method minus graph (m/s)")
        axis.set_title(title, pad=5)
        _panel_label(axis, f"({chr(97 + panel)})")
    axes[1].legend(frameon=False, loc="best")
    figure.tight_layout(w_pad=1.1)
    figure_path = figure_dir / "referee3_tail_controls.pdf"
    _save_figure(figure, figure_path)
    return table_path, figure_path


def _targeting_assets(
    targeting: pd.DataFrame,
    decomposition: pd.DataFrame,
    permuted: pd.DataFrame,
    table_dir: Path,
    figure_dir: Path,
) -> tuple[Path, Path]:
    targeting_required = ("role", "run", "stratum", "model", "n", "share", "rmse")
    decomposition_required = (
        "role",
        "run",
        "stratum",
        "n",
        "shared_mse",
        "between_member_variance",
        "member_mse",
        "identity_residual",
    )
    permuted_required = (
        "role",
        "run",
        "stratum",
        "n",
        "permutation",
        "observed_gel_rmse",
        "permuted_rmse",
    )
    for frame, required, description in (
        (targeting, targeting_required, "targeting-by-stratum"),
        (decomposition, decomposition_required, "error decomposition"),
        (permuted, permuted_required, "permuted gate"),
    ):
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise AssetSchemaError(f"{description} is missing required columns {missing}")
    _require_numeric(targeting, ("n", "share", "rmse"), "targeting-by-stratum")
    _require_numeric(
        decomposition,
        (
            "n",
            "shared_mse",
            "between_member_variance",
            "member_mse",
            "identity_residual",
        ),
        "error decomposition",
    )
    _require_numeric(
        permuted,
        ("n", "permutation", "observed_gel_rmse", "permuted_rmse"),
        "permuted gate",
    )
    _require_unique(targeting, ("role", "run", "stratum", "model"), "targeting")
    _require_unique(decomposition, ("role", "run", "stratum"), "decomposition")
    _require_unique(
        permuted, ("role", "run", "stratum", "permutation"), "permuted gate"
    )
    if float(decomposition["identity_residual"].abs().max()) > 1e-7:
        raise AssetSchemaError(
            "error decomposition violates member MSE = shared MSE + between-member variance"
        )
    if bool(~targeting["share"].between(0.0, 1.0).all()):
        raise AssetSchemaError("targeting stratum shares must lie in [0, 1]")

    evaluation_targeting = targeting[~targeting["role"].eq("calibration")].copy()
    evaluation_decomposition = decomposition[~decomposition["role"].eq("calibration")].copy()
    evaluation_permuted = permuted[~permuted["role"].eq("calibration")].copy()
    if any(frame.empty for frame in (evaluation_targeting, evaluation_decomposition, evaluation_permuted)):
        raise AssetSchemaError("targeting assets have no non-calibration rows")
    _require_values(
        evaluation_targeting,
        "model",
        ("graph ensemble", "GEL-Ped"),
        "targeting-by-stratum",
    )
    pair = evaluation_targeting[
        evaluation_targeting["model"].isin(("graph ensemble", "GEL-Ped"))
    ]
    _validate_complete_cross(
        pair,
        ("role", "run", "stratum"),
        "model",
        ("graph ensemble", "GEL-Ped"),
        "targeting-by-stratum",
    )

    wide = pair.pivot(
        index=["role", "run", "stratum"], columns="model", values=["rmse", "share"]
    )
    wide.columns = [f"{quantity}|{model}" for quantity, model in wide.columns]
    wide = wide.reset_index()
    wide["gel_minus_graph"] = wide["rmse|GEL-Ped"] - wide["rmse|graph ensemble"]
    permutation_summary = (
        evaluation_permuted.groupby(["role", "run", "stratum"], sort=False)
        .agg(observed_gel_rmse=("observed_gel_rmse", "first"), permuted_rmse=("permuted_rmse", "mean"))
        .reset_index()
    )
    permutation_summary["permuted_minus_observed"] = (
        permutation_summary["permuted_rmse"] - permutation_summary["observed_gel_rmse"]
    )
    summary = (
        wide.groupby(["role", "stratum"], sort=False)
        .agg(
            share=("share|GEL-Ped", "mean"),
            graph_rmse=("rmse|graph ensemble", "mean"),
            gel_rmse=("rmse|GEL-Ped", "mean"),
            gel_minus_graph=("gel_minus_graph", "mean"),
        )
        .reset_index()
    )
    permutation_macro = (
        permutation_summary.groupby(["role", "stratum"], sort=False)[
            "permuted_minus_observed"
        ]
        .mean()
        .rename("permutation_gap")
        .reset_index()
    )
    decomposition_macro = (
        evaluation_decomposition.groupby(["role", "stratum"], sort=False)[
            ["shared_mse", "between_member_variance", "member_mse"]
        ]
        .mean()
        .reset_index()
    )
    summary = summary.merge(
        permutation_macro, on=["role", "stratum"], validate="one_to_one"
    ).merge(decomposition_macro, on=["role", "stratum"], validate="one_to_one")
    summary["between_share"] = np.divide(
        summary["between_member_variance"],
        summary["member_mse"],
        out=np.zeros(len(summary), dtype=float),
        where=summary["member_mse"].to_numpy(float) > 0.0,
    )

    rows: list[list[str]] = []
    for role in sorted(summary["role"].astype(str).unique()):
        role_frame = summary[summary["role"].eq(role)].set_index("stratum")
        for stratum in STRATUM_ORDER:
            if stratum not in role_frame.index:
                continue
            current = role_frame.loc[stratum]
            rows.append(
                [
                    _tex(role),
                    _tex(stratum),
                    f"{100.0 * current['share']:.1f}\\%",
                    _format_number(current["graph_rmse"]),
                    _format_number(current["gel_rmse"]),
                    _format_number(current["permutation_gap"]),
                    f"{100.0 * current['between_share']:.1f}\\%",
                ]
            )
    table_path = table_dir / "referee3_targeting_decomposition.tex"
    _atomic_text(
        table_path,
        _table(
            caption=(
                "Where the guard acts. Strata are defined by calibration-normalised "
                "ensemble spread; permutation gaps compare the observed routing with "
                "the same weights randomly reassigned within a run."
            ),
            label="tab:r3-targeting-decomposition",
            column_specification="llrrrrr",
            header=(
                "Regime",
                "Spread stratum",
                "Share",
                "Graph RMSE",
                "GEL-Ped RMSE",
                "Permutation gap",
                "Variance share",
            ),
            rows=rows,
            notes=(
                "Positive permutation gaps favour targeted routing. Variance share is "
                "between-member variance divided by mean member MSE."
            ),
        ),
    )

    roles = sorted(summary["role"].astype(str).unique())
    x = np.arange(len(STRATUM_ORDER), dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
    for color, role in zip(SERIES_COLORS, roles, strict=False):
        role_summary = summary[summary["role"].eq(role)].set_index("stratum")
        positions = [index for index, name in enumerate(STRATUM_ORDER) if name in role_summary.index]
        labels = [STRATUM_ORDER[index] for index in positions]
        axes[0].plot(
            positions,
            role_summary.loc[labels, "gel_minus_graph"],
            marker="o",
            color=color,
            label=role,
        )
        axes[1].plot(
            positions,
            role_summary.loc[labels, "permutation_gap"],
            marker="o",
            color=color,
        )
    aggregate = summary.groupby("stratum", sort=False)[
        ["shared_mse", "between_member_variance", "member_mse"]
    ].mean()
    present = [name for name in STRATUM_ORDER if name in aggregate.index]
    denominator = aggregate.loc[present, "member_mse"].to_numpy(float)
    shared = 100.0 * aggregate.loc[present, "shared_mse"].to_numpy(float) / denominator
    between = (
        100.0 * aggregate.loc[present, "between_member_variance"].to_numpy(float) / denominator
    )
    positions = [STRATUM_ORDER.index(name) for name in present]
    axes[2].bar(positions, shared, color=COLORS["light_grey"], label="Shared error")
    axes[2].bar(
        positions,
        between,
        bottom=shared,
        color=COLORS["orange"],
        label="Member variance",
    )
    titles = ("Guard effect", "Targeting control", "Member-error composition")
    ylabels = (
        "GEL-Ped minus graph RMSE (m/s)",
        "Permuted minus observed RMSE (m/s)",
        "Share of member MSE (%)",
    )
    for panel, (axis, title, ylabel) in enumerate(zip(axes, titles, ylabels, strict=True)):
        _style_axis(axis, zero_line=panel < 2)
        _panel_label(axis, f"({chr(97 + panel)})")
        axis.set_title(title, pad=5)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, ("inside", "0--1", "1--2", ">2"))
        axis.set_xlabel("Spread above envelope (IQR)")
    axes[0].legend(frameon=False, loc="best")
    axes[2].legend(frameon=False, loc="best")
    axes[2].set_ylim(0.0, 105.0)
    figure.tight_layout(w_pad=0.75)
    figure_path = figure_dir / "referee3_targeting_decomposition.pdf"
    _save_figure(figure, figure_path)
    return table_path, figure_path


def _heldout_assets(
    metrics: pd.DataFrame,
    inference: pd.DataFrame,
    table_dir: Path,
    figure_dir: Path,
) -> tuple[Path, Path]:
    metric_columns = (
        "run",
        "role",
        "model",
        "n",
        "rmse",
        "q95",
        "cvar95",
        "rate_error_gt_0_5",
    )
    inference_columns = (
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
    if list(metrics.columns) != list(metric_columns):
        raise AssetSchemaError(
            "heldout primary metrics columns/order differ from the agreed frozen contract"
        )
    if list(inference.columns) != list(inference_columns):
        raise AssetSchemaError(
            "heldout confirmatory inference columns/order differ from the agreed frozen contract"
        )
    _require_numeric(
        metrics, ("n", "rmse", "q95", "cvar95", "rate_error_gt_0_5"), "heldout metrics"
    )
    _require_numeric(
        inference,
        (
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
        ),
        "heldout inference",
    )
    _require_unique(metrics, ("run", "model"), "heldout metrics")
    _require_unique(inference, ("endpoint",), "heldout inference")
    _require_values(metrics, "role", ("untouched T-junction",), "heldout metrics", exact=True)
    _require_values(metrics, "model", ("graph", "guard"), "heldout metrics", exact=True)
    _require_values(
        inference, "endpoint", HELDOUT_ENDPOINTS, "heldout inference", exact=True
    )
    _validate_complete_cross(
        metrics,
        ("run",),
        "model",
        ("graph", "guard"),
        "heldout metrics",
    )
    observed_runs = metrics["run"].nunique()
    if not bool(inference["n_runs"].eq(observed_runs).all()):
        raise AssetSchemaError(
            "heldout inference n_runs does not equal the run count in primary metrics"
        )
    if bool((inference["graph_run_macro"] <= 0.0).any()):
        raise AssetSchemaError("heldout graph run-macro values must be positive")
    if bool((inference["ci95_lower"] > inference["ci95_upper"]).any()):
        raise AssetSchemaError("heldout inference contains a reversed confidence interval")

    labels = {
        "rmse": "RMSE (m/s)",
        "q95": "$q_{.95}$ (m/s)",
        "cvar95": "CVaR$_{.95}$ (m/s)",
        "rate_error_gt_0_5": "$e>0.5$ (share)",
    }
    ordered = inference.set_index("endpoint").loc[list(HELDOUT_ENDPOINTS)]
    rows = []
    for endpoint, current in ordered.iterrows():
        digits = 4
        rows.append(
            [
                labels[endpoint],
                _format_number(current["graph_run_macro"], digits),
                _format_number(current["guard_run_macro"], digits),
                _format_number(current["mean_guard_minus_graph"], digits),
                (
                    f"[{_format_number(current['ci95_lower'], digits)}, "
                    f"{_format_number(current['ci95_upper'], digits)}]"
                ),
                f"{current['relative_mean_difference_percent']:.1f}\\%",
                _format_p(current["holm_adjusted_p"]),
            ]
        )
    table_path = table_dir / "referee3_heldout_tjunction.tex"
    _atomic_text(
        table_path,
        _table(
            caption=(
                "One-shot untouched T-junction evaluation. Inference is paired by "
                "complete physical run and adjusted across the four locked endpoints."
            ),
            label="tab:r3-heldout-tjunction",
            column_specification="lrrrrrr",
            header=(
                "Endpoint",
                "Graph",
                "GEL-Ped",
                "$\\Delta$",
                "95\\% CI",
                "Relative $\\Delta$",
                "Holm $p$",
            ),
            rows=rows,
            notes=(
                "Negative differences favour GEL-Ped. The archive, inclusion rules, "
                "models, guard, and endpoints were frozen before labelled inspection."
            ),
        ),
    )

    relative_low = 100.0 * ordered["ci95_lower"] / ordered["graph_run_macro"]
    relative_high = 100.0 * ordered["ci95_upper"] / ordered["graph_run_macro"]
    relative_mean = ordered["relative_mean_difference_percent"].to_numpy(float)
    y = np.arange(len(ordered), dtype=float)
    figure, axis = plt.subplots(figsize=(4.9, 2.35))
    axis.hlines(
        y,
        relative_low.to_numpy(float),
        relative_high.to_numpy(float),
        color=COLORS["navy"],
        linewidth=1.25,
        zorder=2,
    )
    axis.vlines(
        relative_low.to_numpy(float), y - 0.07, y + 0.07, color=COLORS["navy"]
    )
    axis.vlines(
        relative_high.to_numpy(float), y - 0.07, y + 0.07, color=COLORS["navy"]
    )
    axis.scatter(relative_mean, y, color=COLORS["navy"], s=18, zorder=3)
    axis.axvline(0.0, color="#3A3A3A", linewidth=0.75, zorder=1)
    axis.grid(axis="x", color="#E7E9EB", linewidth=0.55, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_yticks(y, ("RMSE", "$q_{.95}$", "CVaR$_{.95}$", "$e>0.5$"))
    axis.set_xlabel("GEL-Ped minus graph (% of graph run-macro value)")
    axis.set_title("Untouched T-junction: locked confirmatory endpoints", pad=6)
    axis.invert_yaxis()
    figure.tight_layout()
    figure_path = figure_dir / "referee3_heldout_tjunction.pdf"
    _save_figure(figure, figure_path)
    return table_path, figure_path


def _planner_assets(
    frame: pd.DataFrame, table_dir: Path, figure_dir: Path
) -> tuple[Path, Path]:
    required = (
        "role",
        "run",
        "policy",
        "completed",
        "travel_time_seconds",
        "progress_fraction",
        "intrusion_rate",
        "minimum_separation_m",
        "stop_frequency",
        "infeasible_count",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AssetSchemaError(f"planner metrics is missing required columns {missing}")
    _require_numeric(
        frame,
        (
            "progress_fraction",
            "intrusion_rate",
            "minimum_separation_m",
            "stop_frequency",
            "infeasible_count",
        ),
        "planner metrics",
    )
    travel = pd.to_numeric(frame["travel_time_seconds"], errors="coerce")
    if bool((travel.notna() & ~np.isfinite(travel)).any()):
        raise AssetSchemaError("planner travel_time_seconds contains infinite values")
    _require_unique(frame, ("role", "run", "policy"), "planner metrics")
    for rate in ("progress_fraction", "intrusion_rate", "stop_frequency"):
        if bool(~frame[rate].between(0.0, 1.0).all()):
            raise AssetSchemaError(f"planner {rate} must lie in [0, 1]")
    _require_values(frame, "policy", PLANNER_POLICIES, "planner metrics", exact=True)
    _validate_complete_cross(
        frame,
        ("role", "run"),
        "policy",
        PLANNER_POLICIES,
        "planner metrics",
    )
    completed = frame["completed"].astype(str).str.lower().map(
        {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
    )
    if bool(completed.isna().any()):
        raise AssetSchemaError("planner completed must contain booleans")
    frame = frame.assign(_completed=completed, _travel_time=travel)
    summary = (
        frame.groupby(["role", "policy"], sort=False)
        .agg(
            completion=("_completed", "mean"),
            travel_time=("_travel_time", "mean"),
            progress=("progress_fraction", "mean"),
            intrusion_rate=("intrusion_rate", "mean"),
            minimum_separation=("minimum_separation_m", "mean"),
            stop_frequency=("stop_frequency", "mean"),
        )
        .reset_index()
    )
    policy_labels = {
        PLANNER_POLICIES[0]: "Direct + fixed margin",
        PLANNER_POLICIES[1]: "GEL-Ped + fixed margin",
        PLANNER_POLICIES[2]: "GEL-Ped + adaptive margin",
    }
    rows: list[list[str]] = []
    for role in sorted(summary["role"].astype(str).unique()):
        role_frame = summary[summary["role"].eq(role)].set_index("policy")
        for policy in PLANNER_POLICIES:
            current = role_frame.loc[policy]
            rows.append(
                [
                    _tex(role),
                    _tex(policy_labels[policy]),
                    f"{100.0 * current['completion']:.0f}\\%",
                    _format_number(current["travel_time"], 2),
                    _format_number(current["intrusion_rate"], 3),
                    _format_number(current["minimum_separation"], 3),
                    _format_number(current["stop_frequency"], 3),
                ]
            )
    table_path = table_dir / "referee3_planner_utility.tex"
    _atomic_text(
        table_path,
        _table(
            caption=(
                "Paired reactive-planner replay on identical recorded crowd motion "
                "and routes. The replay tests operational consequence, not reciprocal "
                "human response."
            ),
            label="tab:r3-planner-utility",
            column_specification="llrrrrr",
            header=(
                "Regime",
                "Policy",
                "Complete",
                "Travel (s)",
                "Intrusion rate",
                "Min. sep. (m)",
                "Stop rate",
            ),
            rows=rows,
            notes=(
                "Travel time is averaged over completed replays. Intrusions use the "
                "predeclared 0.55-m personal-space threshold."
            ),
        ),
    )

    base = frame[frame["policy"].eq(PLANNER_POLICIES[0])].set_index(["role", "run"])
    comparisons = PLANNER_POLICIES[1:]
    comparison_labels = ("Guarded, fixed", "Guarded, adaptive")
    endpoints = (
        ("minimum_separation_m", "Change in minimum separation (m)"),
        ("intrusion_rate", "Change in intrusion rate"),
        ("_travel_time", "Change in travel time (s)"),
    )
    roles = sorted(frame["role"].astype(str).unique())
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), sharey=True)
    y = np.arange(len(roles), dtype=float)
    for panel, (axis, (endpoint, xlabel)) in enumerate(zip(axes, endpoints, strict=True)):
        for color, marker, offset, policy, label in zip(
            (COLORS["navy"], COLORS["orange"]),
            ("o", "s"),
            (-0.09, 0.09),
            comparisons,
            comparison_labels,
            strict=True,
        ):
            current = frame[frame["policy"].eq(policy)].set_index(["role", "run"])
            if set(current.index) != set(base.index):
                missing_keys = sorted(set(base.index) ^ set(current.index))[:5]
                raise AssetSchemaError(
                    f"planner policy {policy} does not match direct-policy runs: {missing_keys}"
                )
            current = current.reindex(base.index)
            difference = current[endpoint] - base[endpoint]
            macro = difference.groupby(level="role").mean()
            axis.scatter(
                [macro.loc[role] for role in roles],
                y + offset,
                color=color,
                marker=marker,
                s=22,
                label=label,
                zorder=3,
            )
        axis.axvline(0.0, color="#3A3A3A", linewidth=0.75, zorder=1)
        axis.grid(axis="x", color="#E7E9EB", linewidth=0.55, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlabel(xlabel)
        axis.set_yticks(y, roles)
        _panel_label(axis, f"({chr(97 + panel)})")
    axes[0].legend(frameon=False, loc="best")
    figure.tight_layout(w_pad=0.75)
    figure_path = figure_dir / "referee3_planner_utility.pdf"
    _save_figure(figure, figure_path)
    return table_path, figure_path


def _transformer_table(frame: pd.DataFrame, table_dir: Path) -> Path:
    required = (
        "regime",
        "model",
        "runs",
        "rmse_mps",
        "q95_mps",
        "cvar95_mps",
        "guard_firing_rate",
        "mean_transformer_weight",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise AssetSchemaError(f"Transformer summary is missing required columns {missing}")
    _require_numeric(frame, required[2:], "Transformer summary")
    _require_unique(frame, ("regime", "model"), "Transformer summary")
    models = (
        "transformer ensemble",
        "feature-based HGB fallback",
        "guarded transformer",
    )
    _require_values(frame, "model", models, "Transformer summary", exact=True)
    _validate_complete_cross(
        frame, ("regime",), "model", models, "Transformer summary"
    )
    rows: list[list[str]] = []
    for regime in sorted(frame["regime"].astype(str).unique()):
        current = frame[frame["regime"].eq(regime)].set_index("model")
        for model in models:
            row = current.loc[model]
            rows.append(
                [
                    _tex(regime),
                    _tex(MODEL_LABELS[model]),
                    str(int(row["runs"])),
                    _format_number(row["rmse_mps"]),
                    _format_number(row["q95_mps"]),
                    _format_number(row["cvar95_mps"]),
                    f"{100.0 * row['guard_firing_rate']:.1f}\\%",
                ]
            )
    path = table_dir / "referee3_transformer_generalization.tex"
    _atomic_text(
        path,
        _table(
            caption=(
                "Backbone generalisation: the frozen guard applied to a compact "
                "interaction Transformer trained and calibrated under the same run-wise protocol."
            ),
            label="tab:r3-transformer",
            column_specification="llrrrrr",
            header=(
                "Regime",
                "Method",
                "Runs",
                "RMSE",
                "$q_{.95}$",
                "CVaR$_{.95}$",
                "Firing",
            ),
            rows=rows,
            notes="All velocity-error quantities are in m/s.",
        ),
    )
    return path


def _runtime_table(
    table_dir: Path,
    *,
    planner_latency: pd.DataFrame | None,
    training_manifest: dict[str, Any] | None,
    transformer_manifest: dict[str, Any] | None,
    config: dict[str, Any] | None,
) -> Path | None:
    rows: list[list[str]] = []
    primary_size = 5
    if config is not None:
        try:
            primary_size = int(config["replication"]["primary_ensemble_size"])
        except (KeyError, TypeError, ValueError) as error:
            raise AssetSchemaError(
                "config lacks replication.primary_ensemble_size"
            ) from error

    if training_manifest is not None:
        fits = training_manifest.get("fits")
        if not isinstance(fits, list) or not fits:
            raise AssetSchemaError("training manifest has no non-empty fits list")
        try:
            parameter_counts = {int(row["parameters"]) for row in fits}
        except (KeyError, TypeError, ValueError) as error:
            raise AssetSchemaError("training manifest fits lack integer parameters") from error
        if len(parameter_counts) != 1:
            raise AssetSchemaError(
                f"graph fits report inconsistent parameter counts: {sorted(parameter_counts)}"
            )
        parameters = parameter_counts.pop()
        rows.extend(
            [
                ["Graph member", f"{parameters:,}", "--", "--", "Trainable weights"],
                [
                    f"{primary_size}-member graph ensemble",
                    f"{primary_size * parameters:,}",
                    "--",
                    "--",
                    "No weight sharing",
                ],
            ]
        )

    if transformer_manifest is not None:
        fits = transformer_manifest.get("fits")
        seeds = transformer_manifest.get("seeds")
        if not isinstance(fits, list) or not fits or not isinstance(seeds, list) or not seeds:
            raise AssetSchemaError("Transformer manifest lacks non-empty fits/seeds lists")
        try:
            parameter_counts = {int(row["parameter_count"]) for row in fits}
        except (KeyError, TypeError, ValueError) as error:
            raise AssetSchemaError(
                "Transformer manifest fits lack integer parameter_count"
            ) from error
        if len(parameter_counts) != 1:
            raise AssetSchemaError(
                "Transformer fits report inconsistent parameter counts"
            )
        parameters = parameter_counts.pop()
        rows.extend(
            [
                ["Transformer member", f"{parameters:,}", "--", "--", "Trainable weights"],
                [
                    f"{len(seeds)}-member Transformer ensemble",
                    f"{len(seeds) * parameters:,}",
                    "--",
                    "--",
                    "No weight sharing",
                ],
            ]
        )

    if planner_latency is not None:
        required = (
            "role",
            "run",
            "stage",
            "scope",
            "pedestrians",
            "warmup",
            "repeats",
            "mean_ms",
            "median_ms",
            "p95_ms",
            "minimum_ms",
            "maximum_ms",
        )
        missing = [column for column in required if column not in planner_latency.columns]
        if missing:
            raise AssetSchemaError(f"planner latency is missing required columns {missing}")
        _require_numeric(
            planner_latency,
            (
                "pedestrians",
                "warmup",
                "repeats",
                "mean_ms",
                "median_ms",
                "p95_ms",
                "minimum_ms",
                "maximum_ms",
            ),
            "planner latency",
        )
        _require_unique(planner_latency, ("role", "run", "stage"), "planner latency")
        for stage, group in planner_latency.groupby("stage", sort=False):
            scopes = group["scope"].dropna().astype(str).unique()
            if len(scopes) != 1:
                raise AssetSchemaError(f"planner latency stage {stage} has mixed scopes")
            rows.append(
                [
                    _tex(stage),
                    "--",
                    _format_number(group["median_ms"].median(), 3),
                    _format_number(group["p95_ms"].median(), 3),
                    "Post-prediction CPU",
                ]
            )
    if not rows:
        return None
    path = table_dir / "referee3_runtime_parameters.tex"
    _atomic_text(
        path,
        _table(
            caption=(
                "Implementation footprint and latency. Parameter counts and timings "
                "are kept in separate rows because saved-array timing excludes model forward passes."
            ),
            label="tab:r3-runtime-parameters",
            column_specification="lrrrl",
            header=("Component", "Parameters", "Median (ms)", "$p_{.95}$ (ms)", "Scope"),
            rows=rows,
            notes=(
                "Latency is the median across complete-run timing records for each stage; "
                "each record uses its declared warm-up and repeated calls."
            ),
        ),
    )
    return path


def _figure_snippets(figure_dir: Path, table_dir: Path) -> Path:
    definitions = (
        (
            "referee3_draw_stability.pdf",
            "Ensemble-size and member-draw stability. Points are run--draw macro effects; bars are crossed 95\\% resampling intervals. Negative values favour GEL-Ped.",
            "fig:r3-draw-stability",
        ),
        (
            "referee3_tail_controls.pdf",
            "Tail-error comparison against the direct graph ensemble. Each point is a complete-run macro difference for one matched routing rule; negative values favour the comparison method.",
            "fig:r3-tail-controls",
        ),
        (
            "referee3_targeting_decomposition.pdf",
            "Guard targeting and error decomposition across calibration-normalised spread strata. The permutation control preserves the same run-wise weights but breaks their alignment with individual forecasts.",
            "fig:r3-targeting-decomposition",
        ),
        (
            "referee3_heldout_tjunction.pdf",
            "Locked confirmatory effects in the untouched T-junction. Intervals use complete physical runs; effects are normalised only for this visual comparison across endpoint units.",
            "fig:r3-heldout-tjunction",
        ),
        (
            "referee3_planner_utility.pdf",
            "Paired open-loop planner consequences relative to direct graph forecasts with a fixed margin. Pedestrian motion and robot routes are identical within each comparison.",
            "fig:r3-planner-utility",
        ),
    )
    lines = ["% Generated by experiments/referee3_publication_assets.py."]
    for filename, caption, label in definitions:
        if not (figure_dir / filename).exists():
            continue
        lines.extend(
            [
                r"\begin{figure}[!htbp]",
                r"\centering",
                rf"\includegraphics[width=0.99\textwidth]{{{filename}}}",
                rf"\caption{{{caption}}}",
                rf"\label{{{label}}}",
                r"\end{figure}",
                "",
            ]
        )
    path = table_dir / "referee3_figure_snippets.tex"
    _atomic_text(path, "\n".join(lines))
    return path


def build_assets(inputs: AssetInputs, output_root: Path) -> AssetOutputs:
    """Validate frozen outputs and generate deterministic manuscript assets."""

    _configure_matplotlib()
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"

    draw = _read_csv(inputs.draw_inference, (), "draw inference")
    primary = _read_csv(inputs.primary_metrics, (), "primary run metrics")
    targeting = _read_csv(inputs.targeting, (), "targeting-by-stratum")
    decomposition = _read_csv(inputs.decomposition, (), "error decomposition")
    permuted = _read_csv(inputs.permuted_gate, (), "permuted-gate analysis")
    heldout_metrics = _read_csv(
        inputs.heldout_metrics,
        (),
        "heldout primary run metrics",
    )
    heldout_inference = _read_csv(
        inputs.heldout_inference,
        (),
        "heldout confirmatory inference",
    )
    planner = _read_csv(inputs.planner_metrics, (), "planner metrics")

    tables: list[Path] = []
    figures: list[Path] = []
    for builder, arguments in (
        (_draw_assets, (draw, table_dir, figure_dir)),
        (_primary_assets, (primary, table_dir, figure_dir)),
        (
            _targeting_assets,
            (targeting, decomposition, permuted, table_dir, figure_dir),
        ),
        (
            _heldout_assets,
            (heldout_metrics, heldout_inference, table_dir, figure_dir),
        ),
        (_planner_assets, (planner, table_dir, figure_dir)),
    ):
        table_path, figure_path = builder(*arguments)
        tables.append(table_path)
        figures.append(figure_path)

    optional_inputs: list[Path] = []
    if inputs.transformer_summary is not None:
        transformer = _read_csv(
            inputs.transformer_summary, (), "Transformer regime summary"
        )
        tables.append(_transformer_table(transformer, table_dir))
        optional_inputs.append(inputs.transformer_summary)

    latency = None
    if inputs.planner_latency is not None:
        latency = _read_csv(inputs.planner_latency, (), "planner latency")
        optional_inputs.append(inputs.planner_latency)
    training_manifest = None
    if inputs.training_manifest is not None:
        training_manifest = _read_json(inputs.training_manifest, "training manifest")
        optional_inputs.append(inputs.training_manifest)
    transformer_manifest = None
    if inputs.transformer_manifest is not None:
        transformer_manifest = _read_json(
            inputs.transformer_manifest, "Transformer manifest"
        )
        optional_inputs.append(inputs.transformer_manifest)
    config = None
    if inputs.config is not None:
        config = _read_json(inputs.config, "referee-3 configuration")
        optional_inputs.append(inputs.config)
    runtime = _runtime_table(
        table_dir,
        planner_latency=latency,
        training_manifest=training_manifest,
        transformer_manifest=transformer_manifest,
        config=config,
    )
    if runtime is not None:
        tables.append(runtime)

    snippets = _figure_snippets(figure_dir, table_dir)
    required_inputs = [
        inputs.draw_inference,
        inputs.primary_metrics,
        inputs.targeting,
        inputs.decomposition,
        inputs.permuted_gate,
        inputs.heldout_metrics,
        inputs.heldout_inference,
        inputs.planner_metrics,
    ]
    all_outputs = [*tables, *figures, snippets]
    manifest_path = output_root / "referee3_publication_assets_manifest.json"
    manifest = {
        "schema_version": 1,
        "scope": "presentation-only; no model fitting or endpoint selection",
        "inputs": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in [*required_inputs, *optional_inputs]
        ],
        "outputs": [
            {
                "path": str(path.relative_to(output_root)),
                "sha256": _sha256(path),
            }
            for path in all_outputs
        ],
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return AssetOutputs(tuple(tables), tuple(figures), snippets, manifest_path)


def _optional_path(explicit: Path | None, default: Path) -> Path | None:
    if explicit is not None:
        return _require_file(explicit, "optional publication input")
    return default if default.is_file() else None


def _parse_args() -> argparse.Namespace:
    project = project_root()
    processed = project / "data/processed"
    generated = project / "reports/generated"
    parser = argparse.ArgumentParser(
        description="Build strict, result-driven referee-3 publication assets."
    )
    parser.add_argument(
        "--draw-inference",
        type=Path,
        default=processed / "referee3_draw_run_inference.csv",
    )
    parser.add_argument(
        "--primary-metrics",
        type=Path,
        default=processed / "referee3_primary_run_metrics.csv",
    )
    parser.add_argument(
        "--targeting",
        type=Path,
        default=processed / "referee3_targeting_by_stratum.csv",
    )
    parser.add_argument(
        "--decomposition",
        type=Path,
        default=processed / "referee3_error_decomposition.csv",
    )
    parser.add_argument(
        "--permuted-gate",
        type=Path,
        default=processed / "referee3_permuted_gate.csv",
    )
    parser.add_argument(
        "--heldout-metrics",
        type=Path,
        default=generated / "referee3_heldout_primary_run_metrics.csv",
    )
    parser.add_argument(
        "--heldout-inference",
        type=Path,
        default=generated / "referee3_heldout_confirmatory_inference.csv",
    )
    parser.add_argument(
        "--planner-metrics",
        type=Path,
        default=processed / "referee3_planner_metrics.csv",
    )
    parser.add_argument("--transformer-summary", type=Path)
    parser.add_argument("--planner-latency", type=Path)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--transformer-manifest", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project / "manuscript/overleaf_trc",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    project = project_root()
    processed = project / "data/processed"
    inputs = AssetInputs(
        draw_inference=args.draw_inference,
        primary_metrics=args.primary_metrics,
        targeting=args.targeting,
        decomposition=args.decomposition,
        permuted_gate=args.permuted_gate,
        heldout_metrics=args.heldout_metrics,
        heldout_inference=args.heldout_inference,
        planner_metrics=args.planner_metrics,
        transformer_summary=_optional_path(
            args.transformer_summary,
            processed / "referee3_transformer_regime_summary.csv",
        ),
        planner_latency=_optional_path(
            args.planner_latency, processed / "referee3_planner_latency.csv"
        ),
        training_manifest=_optional_path(
            args.training_manifest, processed / "referee3_training_manifest.json"
        ),
        transformer_manifest=_optional_path(
            args.transformer_manifest,
            processed / "referee3_transformer_manifest.json",
        ),
        config=_optional_path(args.config, project / "configs/referee3_lock.json"),
    )
    outputs = build_assets(inputs, args.output_root)
    print(
        json.dumps(
            {
                "status": "ok",
                "tables": [str(path) for path in outputs.tables],
                "figures": [str(path) for path in outputs.figures],
                "manifest": str(outputs.manifest),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
