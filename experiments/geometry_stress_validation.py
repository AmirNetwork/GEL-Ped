# Author: Amir Ghorbani
"""Final evaluation on altered-length and altered-exit corridor runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pedgeom.calibration import (
    DirectionSpeedModel,
    FittedVelocityModel,
    build_velocity_samples,
    fit_tensor_geometry_model,
    velocity_metrics,
)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    processed = project / "data" / "processed"
    raw = project / "data" / "raw" / "2013bidirectional"
    parameters = json.loads((processed / "fitted_velocity_models.json").read_text(encoding="utf-8"))
    splits = json.loads((project / "data" / "splits.json").read_text(encoding="utf-8"))
    interaction_range = parameters["interaction_range_m"]

    models = {}
    for name in [
        "persistence",
        "isotropic_social_force",
        "anisotropic_geometry",
        "decoupled_geometry",
    ]:
        coefficients = parameters["models"][name]
        if name == "decoupled_geometry":
            field = coefficients["field"]
            models[name] = DirectionSpeedModel(
                FittedVelocityModel(tuple(field), np.array(list(field.values()))),
                np.array(coefficients["speed"]),
            )
        else:
            models[name] = FittedVelocityModel(
                tuple(coefficients), np.array(list(coefficients.values()))
            )
    training = [
        build_velocity_samples(raw / f"{run}.txt", interaction_range)
        for run in splits["calibration"]
    ]
    models["tensor_geometry"] = fit_tensor_geometry_model(training)
    rows = []
    batches = []
    for run in splits["held_out_geometry_stress"]:
        print(f"building never-seen geometry run {run}", flush=True)
        batch = build_velocity_samples(raw / f"{run}.txt", interaction_range)
        batches.append(batch)
        for name, model in models.items():
            rows.append({"model": name, "run": run, **velocity_metrics(batch.target, model.predict(batch))})
    results = pd.DataFrame(rows)
    results.to_csv(processed / "geometry_stress_metrics.csv", index=False)

    summary = results.groupby("model", sort=False).agg(
        mean_vector_rmse=("vector_rmse_mps", "mean"),
        mean_displacement_mae=("displacement_mae_m", "mean"),
        mean_angle=("median_angle_deg", "mean"),
        mean_r2=("vector_r2", "mean"),
    )
    summary.to_csv(processed / "geometry_stress_summary.csv")

    pivot = results.pivot(index="run", columns="model", values="vector_rmse_mps")
    improvement = 100.0 * (
        pivot["persistence"] - pivot["tensor_geometry"]
    ) / pivot["persistence"]
    rng = np.random.default_rng(20260722)
    bootstrap = np.mean(rng.choice(improvement.to_numpy(), size=(100_000, len(improvement))), axis=1)
    statistics = {
        "per_run_percent_rmse_improvement": improvement.to_dict(),
        "mean_percent_rmse_improvement": float(improvement.mean()),
        "run_bootstrap_95_percent_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "runs_improved": int((improvement > 0).sum()),
        "runs_total": int(len(improvement)),
    }
    (processed / "geometry_stress_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )

    figure, axis = plt.subplots(figsize=(7.8, 4.4))
    x = np.arange(len(pivot))
    for name, color, marker in (
        [
            ("persistence", "#94a3b8", "o"),
            ("isotropic_social_force", "#2563eb", "s"),
            ("anisotropic_geometry", "#dc2626", "D"),
            ("decoupled_geometry", "#059669", "^"),
            ("tensor_geometry", "#0891b2", "P"),
        ]
    ):
        axis.plot(
            x,
            pivot[name],
            marker=marker,
            linewidth=2,
            markersize=7,
            label=name.replace("_", " "),
            color=color,
        )
    axis.set(
        ylabel="0.4 s vector RMSE (m/s)",
        xlabel="never-seen altered-geometry run",
        xticks=x,
        xticklabels=[name[-4:] for name in pivot.index],
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(fontsize=8)
    axis.set_title("Generalization without refitting")
    figure.tight_layout()
    figure.savefig(project / "figures" / "geometry_stress_validation.png", dpi=240)
    plt.close(figure)
    print("\nGeometry-stress summary\n", summary.to_string(), flush=True)
    print("\nImprovement statistics\n", json.dumps(statistics, indent=2), flush=True)


if __name__ == "__main__":
    main()
