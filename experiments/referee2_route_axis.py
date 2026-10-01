# Author: Amir Ghorbani
"""Route-axis and all-speed audits for the second referee revision.

The primary Juelich representation uses the entry-side route axis.  This
script repeats the complete-run calibration with a strictly causal
prior-velocity axis, then evaluates both encodings on the same untouched
runs.  It also evaluates the locked entry-axis model without the benchmark's
3 m/s target-speed exclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from causal_anchor_revision import build_julich, load_inputs, project_root, settings
from referee2_revision import (
    _fit_mahalanobis,
    _fit_models,
    calibrate,
    calibration_payload,
    evaluate,
    make_oof,
)


def run_axis_audit(*, epochs: int, reuse_oof: bool) -> None:
    project = project_root()
    processed = project / "data" / "processed"
    config, splits = load_inputs(project)
    _, hgb, prospective = settings(project)
    seeds = [config["seed"] + 997 * index for index in range(5)]

    prior_config = dict(config)
    prior_config["primary_goal_method"] = "prior_velocity"
    prior_training_map, prior_datasets = build_julich(
        project, prior_config, splits, history_frames=20
    )
    prior_training = list(prior_training_map.values())
    checkpoint = processed / "referee2_prior_axis_oof.csv"
    if reuse_oof and checkpoint.exists():
        prior_oof = pd.read_csv(checkpoint)
    else:
        prior_oof = make_oof(
            prior_training,
            seeds,
            epochs,
            hgb,
            prospective,
            checkpoint,
        )
    prior_calibrations, prior_gate = calibrate(prior_oof)
    prior_models = _fit_models(prior_training, seeds, epochs, hgb, prospective)
    prior_location, prior_precision = _fit_mahalanobis(prior_training)
    prior_metrics, prior_weights, _, _ = evaluate(
        prior_datasets,
        prior_models,
        prior_calibrations,
        prior_gate,
        prior_location,
        prior_precision,
    )
    prior_metrics.to_csv(processed / "referee2_prior_axis_metrics.csv", index=False)
    prior_weights.to_csv(processed / "referee2_prior_axis_weights.csv", index=False)
    (processed / "referee2_prior_axis_calibration.json").write_text(
        json.dumps(calibration_payload(prior_calibrations), indent=2),
        encoding="utf-8",
    )

    entry_training_map, _ = build_julich(project, config, splits, history_frames=20)
    entry_training = list(entry_training_map.values())
    entry_oof = pd.read_csv(processed / "referee2_oof_calibration.csv")
    entry_calibrations, entry_gate = calibrate(entry_oof)
    entry_models = _fit_models(entry_training, seeds, epochs, hgb, prospective)
    entry_location, entry_precision = _fit_mahalanobis(entry_training)
    _, all_speed_datasets = build_julich(
        project,
        config,
        splits,
        history_frames=20,
        speed_threshold=100.0,
    )
    all_speed_metrics, all_speed_weights, _, _ = evaluate(
        all_speed_datasets,
        entry_models,
        entry_calibrations,
        entry_gate,
        entry_location,
        entry_precision,
    )
    all_speed_metrics.to_csv(
        processed / "referee2_all_speed_metrics.csv", index=False
    )
    all_speed_weights.to_csv(
        processed / "referee2_all_speed_weights.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--reuse-oof", action="store_true")
    args = parser.parse_args()
    run_axis_audit(epochs=args.epochs, reuse_oof=args.reuse_oof)


if __name__ == "__main__":
    main()
