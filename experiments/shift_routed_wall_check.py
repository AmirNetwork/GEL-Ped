# Author: Amir Ghorbani
"""Retrain final GEL-Ped without wall input to audit the crossing comparison."""

from __future__ import annotations

import json

from reviewer_revision_analysis import build_datasets, run_metrics, summarize, without_wall
from major_revision_analysis import load_inputs, project_root
from shift_routed_upgrade import fit_routed_models


def main() -> None:
    project = project_root()
    processed = project / "data" / "processed"
    config, splits = load_inputs(project)
    selected = json.loads(
        (processed / "selected_hyperparameters.json").read_text(encoding="utf-8")
    )
    hgb_selection = json.loads(
        (processed / "shift_routed_hgb_selection.json").read_text(encoding="utf-8")
    )
    training, datasets = build_datasets(project, config, splits)
    wall_free_training = [without_wall(batch) for batch in training.values()]
    wall_free_datasets = {
        name: [without_wall(batch) for batch in batches]
        for name, batches in datasets.items()
    }
    fitted = fit_routed_models(
        wall_free_training, selected, config, hgb_selection
    )
    models = {
        name: fitted[name]
        for name in ("direct_mlp", "direct_routed_control", "gel_ped")
    }
    metrics = run_metrics(wall_free_datasets, models)
    metrics.to_csv(processed / "shift_routed_wall_check_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(processed / "shift_routed_wall_check_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
