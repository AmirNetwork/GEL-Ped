# GEL-Ped

**GEL-Ped** is **Guarded Ensemble Learning for Pedestrians**, authored by
**Amir Ghorbani**. It predicts mean pedestrian velocity over the next 0.4 s
from a 0.8-s observed history.

The final model has three deliberately small parts:

1. five independently initialized graph residual learners whose mean is the
   default forecast;
2. a bounded histogram-gradient-boosted residual model using current
   behaviour-aligned fields; and
3. a local, past-only guard that measures graph-member spread. The graph
   weight is exactly one inside a leave-one-complete-run-out calibration
   envelope and decays only when local spread exceeds that envelope.

The primary method does **not** use prospective closest-approach geometry.
That encoder is retained only as a mechanism control. The paper's claim is
therefore about limiting a detectable neural-instability tail while preserving
the graph ensemble in supported flow, not about universal state-of-the-art
accuracy.

## Main evidence

- Complete physical runs, rather than frames, define every split and every
  inferential unit.
- In perpendicular crossing, GEL-Ped changes mean complete-run RMSE from
  0.3543 to 0.3463 m/s relative to the five-member graph mean. All 13 runs
  improve, but the median difference is only -0.0022 m/s and one unstable run
  contributes substantially to the mean.
- In the highest local-instability stratum, graph RMSE is 0.5654 m/s and
  GEL-Ped RMSE is 0.4665 m/s.
- Coarse HGB is marginally lower than GEL-Ped in crossing (0.3455 versus
  0.3463 m/s) but is 5--8% worse in the other four controlled regimes.
- On all five ETH/UCY leave-one-scene-out tests, the equal-scene macro RMSE is
  0.2267 m/s for the graph ensemble and 0.2262 m/s for GEL-Ped. The difference
  is descriptive and non-significant.
- Recursive crossing audits show smaller graph-to-GEL error at 0.8, 1.6, and
  3.2 s, but coarse HGB and constant velocity are better at 3.2 s. GEL-Ped is
  not presented as a replacement for a dedicated long-horizon predictor.

## Manuscript-to-code map

- sample construction and frame-rate handling:
  `src/pedgeom/calibration.py` and `src/pedgeom/datasets.py`
- graph residual learner: `src/pedgeom/graph_baseline.py`
- behavioural features, boosted models, and controls:
  `src/pedgeom/benchmarks.py`
- ETH/UCY adapter: `src/pedgeom/external.py`
- primary five-seed experiment: `experiments/referee2_revision.py`
- complete Holm family and per-run statistics:
  `experiments/referee2_statistics.py`
- collision, sensitivity, uncertainty, and rollout audits:
  `experiments/referee2_operational_audit.py`
- route-axis and all-speed audits: `experiments/referee2_route_axis.py`
- calibration-only graph capacity sweep:
  `experiments/graph_capacity_sweep.py`
- publication figures: `experiments/referee2_figures.py`
- frozen specification: `configs/referee2_lock.json`
- final manuscript: `manuscript/overleaf_trc/main.tex`

## Installation

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pip install -r requirements-graph.txt
```

The last command installs the CPU build of PyTorch used by the graph model.

## Reproduce the final analysis

Download the Juelich, Tordeux, and ETH/UCY archives documented in
`data/README.md` and place their extracted trajectories in the stated
directories. Raw third-party data are not redistributed.

```powershell
.\.venv\Scripts\python experiments\referee2_revision.py
.\.venv\Scripts\python experiments\referee2_statistics.py
.\.venv\Scripts\python experiments\referee2_operational_audit.py
.\.venv\Scripts\python experiments\referee2_route_axis.py
.\.venv\Scripts\python experiments\graph_capacity_sweep.py
.\.venv\Scripts\python experiments\referee2_figures.py
.\.venv\Scripts\python -m pytest -q
```

The experiment scripts record random seeds, run identifiers, fitted guard
parameters, and complete-run outputs in `data/processed/`.

## Licence and citation

Code is released under the MIT licence. Dataset terms remain with their source
archives. See `CITATION.cff` and cite the original dataset records when using
the controlled or naturalistic experiments.
