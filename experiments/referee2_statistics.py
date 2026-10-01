# Author: Amir Ghorbani
"""Complete run-level statistical report for the second-referee revision."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pedgeom.statistics import exact_paired_randomization_pvalue


PRIMARY = "GEL-Ped instability guard"
COMPARATORS = (
    "five-member graph ensemble",
    "coordinate-median graph ensemble",
    "anchored coarse HGB",
)


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    processed = project / "data/processed"
    metrics = pd.read_csv(processed / "referee2_julich_metrics.csv")
    datasets = tuple(metrics["dataset"].drop_duplicates())
    rows: list[dict] = []
    for dataset in datasets:
        selected = metrics[metrics["dataset"].eq(dataset)]
        pivot = selected.pivot(index="run", columns="model", values="vector_rmse_mps")
        for comparator in COMPARATORS:
            difference = pivot[PRIMARY] - pivot[comparator]
            leave_one_out = (
                [difference.drop(index).mean() for index in difference.index]
                if len(difference) > 1
                else [np.nan]
            )
            rows.append(
                {
                    "dataset": dataset,
                    "primary": PRIMARY,
                    "comparator": comparator,
                    "runs": len(difference),
                    "wins": int(np.sum(difference < 0.0)),
                    "ties": int(np.sum(difference == 0.0)),
                    "mean_difference_mps": float(difference.mean()),
                    "median_difference_mps": float(difference.median()),
                    "minimum_difference_mps": float(difference.min()),
                    "maximum_difference_mps": float(difference.max()),
                    "loo_mean_min_mps": float(np.nanmin(leave_one_out)),
                    "loo_mean_max_mps": float(np.nanmax(leave_one_out)),
                    "relative_difference_percent": float(
                        100.0 * difference.mean() / pivot[comparator].mean()
                    ),
                    "exact_two_sided_p": exact_paired_randomization_pvalue(
                        difference.to_numpy()
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p"] = holm(frame["exact_two_sided_p"].tolist())
    frame.to_csv(processed / "referee2_complete_holm_family.csv", index=False)

    external = pd.read_csv(processed / "referee2_eth_ucy_metrics.csv")
    scene = external.groupby(["dataset", "model"], sort=False)[
        "vector_rmse_mps"
    ].mean().unstack()
    difference = scene[PRIMARY] - scene["five-member graph ensemble"]
    payload = {
        "clusters": "five held-out ETH/UCY scenes",
        "scene_differences_mps": difference.to_dict(),
        "macro_mean_difference_mps": float(difference.mean()),
        "macro_median_difference_mps": float(difference.median()),
        "wins": int(np.sum(difference < 0.0)),
        "exact_two_sided_p": exact_paired_randomization_pvalue(
            difference.to_numpy()
        ),
        "minimum_attainable_two_sided_p": 2.0 / (2 ** len(difference)),
    }
    (processed / "referee2_eth_ucy_statistics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
