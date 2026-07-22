"""Calibrate on frozen runs and evaluate once on untouched Juelich runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pedgeom.calibration import (
    FittedVelocityModel,
    build_velocity_samples,
    fit_velocity_model,
    velocity_metrics,
)


MODEL_FEATURES = {
    "persistence": ("persistence",),
    "goal_only": ("goal",),
    "canonical_field": ("goal", "isotropic", "wall"),
    "isotropic_social_force": ("persistence", "goal", "isotropic", "wall"),
    "anisotropic_geometry": (
        "persistence",
        "goal",
        "isotropic",
        "forward",
        "wall",
    ),
}


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    raw = project / "data" / "raw" / "2013bidirectional"
    splits = json.loads((project / "data" / "splits.json").read_text(encoding="utf-8"))
    ranges = [0.20, 0.35, 0.50, 0.80, 1.20, 1.60]

    cache: dict[tuple[str, float], object] = {}

    def batch(run: str, interaction_range: float):
        key = (run, interaction_range)
        if key not in cache:
            print(f"building {run}, range={interaction_range:.2f} m", flush=True)
            cache[key] = build_velocity_samples(raw / f"{run}.txt", interaction_range)
        return cache[key]

    # Leave-one-run-out selection uses calibration runs only.
    cv_rows = []
    calibration_runs = splits["calibration"]
    for interaction_range in ranges:
        run_batches = {run: batch(run, interaction_range) for run in calibration_runs}
        for held_out in calibration_runs:
            training = [run_batches[run] for run in calibration_runs if run != held_out]
            model = fit_velocity_model(training, MODEL_FEATURES["anisotropic_geometry"])
            metrics = velocity_metrics(
                run_batches[held_out].target, model.predict(run_batches[held_out])
            )
            cv_rows.append({"range_m": interaction_range, "run": held_out, **metrics})
    cv = pd.DataFrame(cv_rows)
    selected_range = float(cv.groupby("range_m")["vector_rmse_mps"].mean().idxmin())
    print(f"selected interaction range: {selected_range:.2f} m", flush=True)

    calibration = [batch(run, selected_range) for run in calibration_runs]
    fitted: dict[str, FittedVelocityModel] = {}
    for name, features in MODEL_FEATURES.items():
        fitted[name] = fit_velocity_model(calibration, features)
        print(name, dict(zip(features, fitted[name].coefficients, strict=True)), flush=True)

    rows = []
    validation_batches = [batch(run, selected_range) for run in splits["held_out_validation"]]
    for model_name, model in fitted.items():
        for validation in validation_batches:
            metrics = velocity_metrics(validation.target, model.predict(validation))
            rows.append({"model": model_name, "run": validation.run, **metrics})
    results = pd.DataFrame(rows)

    output_dir = project / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    cv.to_csv(output_dir / "interaction_range_cross_validation.csv", index=False)
    results.to_csv(output_dir / "heldout_velocity_metrics.csv", index=False)
    parameters = {
        "interaction_range_m": selected_range,
        "horizon_s": 0.4,
        "calibration_runs": calibration_runs,
        "held_out_validation_runs": splits["held_out_validation"],
        "models": {
            name: dict(zip(model.names, model.coefficients.tolist(), strict=True))
            for name, model in fitted.items()
        },
    }
    (output_dir / "fitted_velocity_models.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )

    order = list(MODEL_FEATURES)
    aggregate = results.groupby("model", sort=False).agg(
        mean_vector_rmse=("vector_rmse_mps", "mean"),
        sem_vector_rmse=("vector_rmse_mps", "sem"),
        mean_displacement_mae=("displacement_mae_m", "mean"),
        mean_speed_mae=("speed_mae_mps", "mean"),
        mean_angle=("median_angle_deg", "mean"),
        mean_r2=("vector_r2", "mean"),
    ).reindex(order)
    aggregate.to_csv(output_dir / "heldout_velocity_summary.csv")

    pivot = results.pivot(index="run", columns="model", values="vector_rmse_mps")
    improvements = {}
    rng = np.random.default_rng(20260722)
    for baseline in ["persistence", "isotropic_social_force"]:
        values = 100.0 * (pivot[baseline] - pivot["anisotropic_geometry"]) / pivot[baseline]
        bootstrap = np.mean(rng.choice(values.to_numpy(), size=(100_000, len(values))), axis=1)
        improvements[baseline] = {
            "per_run_percent": values.to_dict(),
            "mean_percent": float(values.mean()),
            "run_bootstrap_95_percent_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "runs_improved": int((values > 0).sum()),
            "runs_total": int(len(values)),
        }
    (output_dir / "heldout_improvement_statistics.json").write_text(
        json.dumps(improvements, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    x = np.arange(len(order))
    labels = [name.replace("_", "\n") for name in order]
    axes[0].bar(
        x,
        aggregate["mean_vector_rmse"],
        yerr=aggregate["sem_vector_rmse"],
        color=["#94a3b8", "#cbd5e1", "#60a5fa", "#2563eb", "#dc2626"],
        capsize=3,
    )
    axes[0].set(ylabel="held-out vector RMSE (m/s)", xticks=x, xticklabels=labels)
    axes[0].grid(axis="y", alpha=0.2)

    for model_name, color in [
        ("persistence", "#94a3b8"),
        ("isotropic_social_force", "#60a5fa"),
        ("anisotropic_geometry", "#dc2626"),
    ]:
        subset = results[results["model"] == model_name]
        axes[1].plot(
            subset["run"].str[-4:],
            subset["vector_rmse_mps"],
            marker="o",
            label=model_name.replace("_", " "),
            color=color,
        )
    axes[1].set(xlabel="untouched experimental run", ylabel="vector RMSE (m/s)")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.suptitle("0.4 s velocity forecast on five frozen Juelich corridor runs")
    figure.tight_layout()
    figure.savefig(project / "figures" / "heldout_velocity_validation.png", dpi=240)
    plt.close(figure)
    print("\nHeld-out run-balanced summary\n", aggregate.to_string(), flush=True)


if __name__ == "__main__":
    main()
