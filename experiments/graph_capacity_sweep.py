# Author: Amir Ghorbani
"""Run-wise calibration-only capacity and epoch sweep for the graph baseline."""

from __future__ import annotations


import pandas as pd

from pedgeom.calibration import velocity_metrics
from pedgeom.graph_baseline import fit_graph_interaction_network

from causal_anchor_revision import build_julich, load_inputs, project_root


CANDIDATES = (
    {"width": 32, "epochs": 25},
    {"width": 48, "epochs": 25},
    {"width": 48, "epochs": 45},
    {"width": 48, "epochs": 75},
    {"width": 96, "epochs": 45},
    {"width": 96, "epochs": 75},
)


def main() -> None:
    project = project_root()
    config, splits = load_inputs(project)
    training_map, _ = build_julich(project, config, splits, history_frames=20)
    training = list(training_map.values())
    rows: list[dict] = []
    output = project / "data/processed/referee2_graph_capacity_sweep.csv"
    for candidate_index, candidate in enumerate(CANDIDATES):
        for held_out, validation in enumerate(training):
            fitting = [batch for index, batch in enumerate(training) if index != held_out]
            model = fit_graph_interaction_network(
                fitting,
                width=candidate["width"],
                epochs=candidate["epochs"],
                seed=config["seed"] + held_out,
                residual=True,
            )
            rows.append(
                {
                    "candidate": candidate_index,
                    **candidate,
                    "held_out_run": validation.run,
                    "parameters": model.parameter_count,
                    **velocity_metrics(
                        validation.target,
                        model.predict(validation, speed_cap=100.0),
                    ),
                }
            )
            pd.DataFrame(rows).to_csv(output, index=False)
            print(
                f"candidate {candidate_index + 1}/{len(CANDIDATES)}, "
                f"fold {held_out + 1}/{len(training)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
