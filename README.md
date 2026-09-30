# GEL-Ped

**GEL-Ped** is **Geometry-guarded Expert Learning for Pedestrian Prediction**,
authored by **Amir Ghorbani**. It predicts 0.4-s pedestrian velocity from a
0.8-s observed history.

The model combines:

1. a permutation-invariant graph network that learns interaction corrections;
2. a prospective-geometry expert built from closest-approach time, clearance,
   closing speed and passing side; and
3. a cross-fitted causal guard that reduces graph weight when the experts
   disagree beyond their calibration envelope.

Both experts learn corrections around observed velocity. The router uses the
current and four preceding sampled frames only; it does not use a future target,
test-regime label, or complete-run aggregate.

## Main evidence

- Five-seed mean RMSE is 0.1607, 0.1542, and 0.3457 m/s in familiar,
  altered-geometry, and perpendicular-crossing tests.
- In crossing, GEL-Ped is 4.84% below the matched graph interaction network and
  8.14% below a direct expert system with the same branches and router. All 13
  complete runs improve; Holm-adjusted p = 0.003662.
- Mean graph weight is about 0.69 in both corridor tests and 0.40 in crossing.
  Buffer disagreement correlates with graph-minus-geometry error in crossing
  (Spearman rho = 0.389, 694 buffers).
- The graph network's crossing seed SD is 0.0195 m/s; GEL-Ped's is 0.0008 m/s.
- On the naturalistic biwi_eth hold-out, the graph network is slightly better
  (0.4101 versus 0.4137 m/s). This is a documented scope boundary.

## Manuscript-to-code map

- sample construction and past-only route estimation:
  src/pedgeom/calibration.py and src/pedgeom/datasets.py
- closest-approach state, causal router and physical baselines:
  src/pedgeom/benchmarks.py
- graph interaction expert: src/pedgeom/graph_baseline.py
- official ETH/UCY adapter: src/pedgeom/external.py
- primary five-seed analysis, ablations, conflict strata, collision screening,
  uncertainty and sensitivity: experiments/causal_anchor_revision.py
- coordinate-noise audit: experiments/measurement_noise_audit.py
- publication figures: experiments/causal_revision_figures.py
- immutable split: data/splits.json
- generated outputs: data/processed/causal_* and data/processed/external_eth_*

## Installation

Python 3.11 or newer is required.

    python -m venv .venv
    .\.venv\Scripts\python -m pip install -e ".[dev]"
    .\.venv\Scripts\python -m pip install -r requirements-graph.txt

The last command installs the CPU build of PyTorch used by the graph model.

## Reproduce the revised experiment

Download the two CC BY 4.0 Juelich archives documented in data/README.md and
place their extracted trajectories in the stated directories. The naturalistic
ETH evaluation reads the official four-column files distributed by
Trajectron++; raw data are not redistributed.

    .\.venv\Scripts\python experiments\causal_anchor_revision.py --seeds 5 --calibration-graph-epochs 25 --graph-epochs 25
    .\.venv\Scripts\python experiments\measurement_noise_audit.py
    .\.venv\Scripts\python experiments\causal_revision_figures.py

All primary settings are frozen in the scripts and calibration JSON. Stochastic
training uses recorded seeds and gives equal aggregate loss weight to each
complete calibration run.

## Licence and citation

Code is released under the MIT licence. Dataset terms remain with their source
archives. See CITATION.cff and cite both Juelich dataset DOI records when using
the controlled experiments.
