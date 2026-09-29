# Author: Amir Ghorbani
"""Retrain final GEL-Ped without wall input to audit the crossing comparison."""

from __future__ import annotations

import json

from pedgeom.benchmarks import calibrate_disagreement_router

from anticipatory_residual_upgrade import fit_hgb
from major_revision_analysis import ModelSpec
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
    prospective_selection = json.loads(
        (processed / "prospective_selection.json").read_text(encoding="utf-8")
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
    parameters = {
        key: prospective_selection[key]
        for key in ("horizon_s", "time_scale_s", "clearance_scale_m")
    }
    residual_settings = {
        key: value
        for key, value in hgb_selection["anchored_residual"].items()
        if key != "candidate"
    }
    direct_settings = {
        key: value
        for key, value in hgb_selection["direct"].items()
        if key != "candidate"
    }
    prospective_residual = fit_hgb(
        wall_free_training, parameters, residual_settings, config["seed"], residual=True
    )
    prospective_direct = fit_hgb(
        wall_free_training, parameters, direct_settings, config["seed"], residual=False
    )
    cap = selected["speed_cap_mps"]
    gel = calibrate_disagreement_router(
        wall_free_training,
        prospective_residual,
        fitted["kinematic_residual_mlp"].model,
    )
    control = calibrate_disagreement_router(
        wall_free_training, prospective_direct, fitted["direct_mlp"].model
    )
    models = {
        name: fitted[name]
        for name in ("direct_mlp", "direct_routed_control", "gel_ped")
    }
    models["prospective_direct_control"] = ModelSpec(control, cap, -1)
    models["gel_ped_anticipatory"] = ModelSpec(gel, cap, -1)
    metrics = run_metrics(wall_free_datasets, models)
    metrics.to_csv(processed / "shift_routed_wall_check_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(processed / "shift_routed_wall_check_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
