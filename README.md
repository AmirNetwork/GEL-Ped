# GEL-Ped

**GEL-Ped** is **Geometry-Encoded Learning for Pedestrian Prediction**, authored by
**Amir Ghorbani**. It is a shift-aware predictor for local crowd motion. Current velocity
anchors two route-aligned residual learners: gradient boosting for calibration-like flow
and a neural residual for shifted flow. Their unlabeled disagreement determines the
deployed mixture.

This repository reproduces the manuscript:
*GEL-Ped: Shift-aware geometry-encoded residual learning for short-term pedestrian
prediction*.

## What the experiments show

- GEL-Ped lowers 0.4 s velocity RMSE by 3.4% in familiar corridors, 5.7% under altered
  geometry, and 4.6% in thirteen perpendicular-crossing runs relative to a direct MLP.
- A direct dual-backbone control uses the same boosted and neural learners and the same
  disagreement router. GEL-Ped remains 4.6% better in crossing, isolating the value of
  residual learning around the kinematic anchor.
- Every crossing run improves; the paired exact test reaches `p = 0.000244` and the
  run-bootstrap interval excludes zero.
- In autoregressive crossing evaluation, displacement error is 17.0% lower at 2.0 s and
  19.9% lower at 3.2 s than the direct MLP.
- With two to seven complete calibration runs, GEL-Ped beats both the direct MLP and
  constant velocity in the regime-balanced low-data analysis.
- Wall-free retraining preserves the crossing advantage, so a missing boundary feature
  does not explain the result.

The crossing archive changes encounter topology but shares the controlled research
infrastructure of the corridor archive. The manuscript calls this a **cross-topology
test**, not naturalistic or multi-site external validation.

## Manuscript-to-code map

- sample construction and past-only fields: `pedgeom.calibration.build_velocity_samples`
- route-aligned inputs and kinematic anchor: `pedgeom.calibration`
- residual experts and disagreement router: `pedgeom.benchmarks`
- final complete-run analysis: `experiments/shift_routed_upgrade.py`
- five-seed sensitivity: `experiments/shift_routed_seed_sensitivity.py`
- attribution, wall, smoothing, route and strata checks:
  `experiments/reviewer_revision_analysis.py`
- autoregressive and conflict evaluation: `experiments/autoregressive_rollout.py`
- final wall-free refit: `experiments/shift_routed_wall_check.py`
- complete-subset low-data study: `experiments/shift_routed_data_efficiency.py`
- publication figures: `experiments/final_publication_figures.py`

## Repository map

- `src/pedgeom/`: models, fields, calibration, metrics and statistics
- `experiments/`: executable primary, sensitivity and figure analyses
- `configs/`: frozen numerical settings
- `tests/`: analytic, data, statistical and regression tests
- `data/splits.json`: immutable complete-run partition
- `data/processed/`: generated run-level results and diagnostics
- `figures/`: publication figures

## Reproduce

Python 3.11 or newer is required. Download the two CC BY 4.0 trajectory archives listed
in `data/README.md`, place them in the documented directories, and run:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.\run_research.ps1
```

The main 6,000-forecast-per-run analysis and dedicated 750-forecast-per-run low-data
analysis are separate by design. All sampling and stochastic fitting use recorded seeds.
Raw trajectory archives are not redistributed.

## Licence and citation

Code is released under the MIT licence. Dataset terms remain with the source archives.
See `CITATION.cff` for software citation metadata and cite both dataset DOI records when
using this package.
