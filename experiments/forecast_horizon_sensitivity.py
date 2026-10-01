# Author: Amir Ghorbani
"""Check whether conclusions persist across forecast horizons."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pedgeom.calibration import (
    build_velocity_samples,
    fit_direction_speed_model,
    fit_tensor_geometry_model,
    fit_velocity_model,
    velocity_metrics,
)


FEATURES = {
    "persistence": ("persistence",),
    "isotropic_social_force": ("persistence", "goal", "isotropic", "wall"),
    "anisotropic_geometry": ("persistence", "goal", "isotropic", "forward", "wall"),
}


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    raw = project / "data" / "raw" / "2013bidirectional"
    splits = json.loads((project / "data" / "splits.json").read_text(encoding="utf-8"))
    rows = []
    for horizon_frames in [5, 10, 20]:
        horizon_s = horizon_frames / 25.0
        print(f"building horizon {horizon_s:.1f} s", flush=True)
        training = [
            build_velocity_samples(
                raw / f"{run}.txt", interaction_range=0.8, horizon_frames=horizon_frames
            )
            for run in splits["calibration"]
        ]
        validation = [
            build_velocity_samples(
                raw / f"{run}.txt", interaction_range=0.8, horizon_frames=horizon_frames
            )
            for run in splits["held_out_validation"]
        ]
        for name, features in FEATURES.items():
            model = fit_velocity_model(training, features)
            for batch in validation:
                rows.append(
                    {
                        "horizon_s": horizon_s,
                        "model": name,
                        "run": batch.run,
                        **velocity_metrics(batch.target, model.predict(batch), horizon_s),
                    }
                )
        model = fit_direction_speed_model(training)
        for batch in validation:
            rows.append(
                {
                    "horizon_s": horizon_s,
                    "model": "decoupled_geometry",
                    "run": batch.run,
                    **velocity_metrics(batch.target, model.predict(batch), horizon_s),
                }
            )
        model = fit_tensor_geometry_model(training)
        for batch in validation:
            rows.append(
                {
                    "horizon_s": horizon_s,
                    "model": "tensor_geometry",
                    "run": batch.run,
                    **velocity_metrics(batch.target, model.predict(batch), horizon_s),
                }
            )
    results = pd.DataFrame(rows)
    processed = project / "data" / "processed"
    results.to_csv(processed / "forecast_horizon_sensitivity.csv", index=False)
    summary = results.groupby(["horizon_s", "model"], sort=False).agg(
        vector_rmse_mps=("vector_rmse_mps", "mean"),
        displacement_mae_m=("displacement_mae_m", "mean"),
        median_angle_deg=("median_angle_deg", "mean"),
        vector_r2=("vector_r2", "mean"),
    )
    summary.to_csv(processed / "forecast_horizon_sensitivity_summary.csv")

    figure, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    colors = {
        "persistence": "#94a3b8",
        "isotropic_social_force": "#2563eb",
        "anisotropic_geometry": "#dc2626",
        "decoupled_geometry": "#059669",
        "tensor_geometry": "#0891b2",
    }
    for name in [*FEATURES, "decoupled_geometry", "tensor_geometry"]:
        model_summary = summary.xs(name, level="model")
        label = name.replace("_", " ")
        axes[0].plot(model_summary.index, model_summary["vector_rmse_mps"], "o-", label=label, color=colors[name])
        axes[1].plot(model_summary.index, model_summary["displacement_mae_m"], "o-", label=label, color=colors[name])
    axes[0].set(xlabel="forecast horizon (s)", ylabel="vector RMSE (m/s)")
    axes[1].set(xlabel="forecast horizon (s)", ylabel="displacement MAE (m)")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Held-out sensitivity to forecast horizon")
    figure.tight_layout()
    figure.savefig(project / "figures" / "forecast_horizon_sensitivity.png", dpi=240)
    plt.close(figure)
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
