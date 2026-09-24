# GEL-Ped

**GEL-Ped** stands for **Geometry-Encoded Learning for Pedestrian Forecasting**. This repository, authored by **Amir Ghorbani**, reproduces the code, experiments, numerical results, and figures for *GEL-Ped: Geometry-encoded learning for pedestrian forecasting under limited data and unseen flow topology*.

The method combines two experts:

1. a direct invariant neural predictor for flexible interpolation; and
2. a geometry-guided neural predictor that starts from a route-aligned longitudinal--lateral forecast and learns only its residual error.

The reported GEL-Ped model is their calibration-selected convex blend. Its machine-readable model identifier is `gel_ped`. The tensor representation is a behavioural geometry and visualization device; it is not a claim that relativistic effects govern walking.

## Practical result

At a 0.4 s forecast horizon, all model choices are made from seven complete corridor calibration runs and then frozen. On untouched runs:

- in familiar corridors, GEL-Ped and the direct network are nearly tied (0.1803 versus 0.1812 m/s RMSE); this small difference is not the basis of the paper's claim;
- on thirteen external 90-degree crossing runs, GEL-Ped improves the direct network by 2.92% (0.3572 versus 0.3680 m/s), a protocol-matched 2024 goal-stable hybrid by 3.04%, and a parameter-matched two-network direct ensemble by 1.49%; all thirteen runs improve in each comparison;
- across 0.2, 0.4, 0.8, and 1.2 s horizons, crossing reductions from the direct network are 1.15%, 2.92%, 3.11%, and 1.32%, with significance retained after correction across horizons;
- after architecture selection is fixed, GEL-Ped lowers pooled error for all 127 possible subsets of one to seven fitting runs, including a 14.49% mean reduction for singleton choices; each fitting run is capped at 750 forecasts to make this a genuine low-data test;
- across all twenty-one untouched runs, it reduces error by 16.77% relative to constant velocity, 14.10% relative to calibrated Social Force, 7.39% relative to unrestricted linear regression, 2.06% relative to the direct network, and 1.03% relative to the equal-size direct ensemble.

The intended decision rule is therefore explicit: a direct network is a reasonable simpler option for well-sampled familiar-scene interpolation; GEL-Ped is the more reliable choice when calibration data are scarce or the deployment topology may change.

## Manuscript-to-code map

The implementation mirrors the manuscript without depending on fragile equation numbering:

- forecast samples and local interaction fields: `pedgeom.calibration.build_velocity_samples`;
- route-aligned longitudinal--lateral response: `pedgeom.calibration.TensorGeometryModel`;
- geometry-guided residual expert: `pedgeom.benchmarks.TensorResidualRegressor`;
- matched direct expert: `pedgeom.benchmarks.InvariantRegressor` and `fit_interaction_mlp`;
- calibration-selected GEL-Ped blend: `pedgeom.benchmarks.GELPedRegressor`;
- calibrated Social Force reference and run-balanced metrics: `pedgeom.calibration`.

## Repository map

- `src/pedgeom/`: fields, tensor model, neural benchmarks, calibration, metrics, and statistics
- `experiments/major_revision_analysis.py`: frozen primary analysis
- `experiments/capacity_matched_neural_ensemble.py`: equal-size two-network control
- `experiments/data_efficiency_analysis.py`: calibration-data sensitivity
- `experiments/neural_horizon_sensitivity.py`: frozen cross-horizon neural transfer
- `experiments/refresh_confirmatory_holm.py`: confirmatory multiplicity update
- `experiments/publication_figures.py`: experiment-context and real-trajectory visuals
- `experiments/publication_upgrade_figures.py`: architecture and result visuals
- `experiments/manuscript_consistency_audit.py`: data and benchmark-fairness audit; it also checks manuscript claims when a manuscript source is present
- `configs/major_revision.json`: machine-readable experiment configuration
- `tests/`: analytic, numerical, data, statistical, and regression tests
- `data/splits.json`: immutable complete-run partition
- `data/processed/`: run-level results, uncertainty, diagnostics, and fitted values
- `figures/`: generated publication figures

## Reproduce

Python 3.11 or newer is required. Download the two CC BY 4.0 trajectory archives described in `data/README.md`, place them in the expected directories, and run:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.\run_research.ps1
```

The runner executes the tests and lint checks, rebuilds all primary and sensitivity results, regenerates the figures, and writes the machine-readable audit. The submission manuscript is distributed separately from this public code repository.

The frozen stochastic seed is `20260722`. Raw trajectory archives are not redistributed; processed result summaries contain no substitute copy of the source trajectories.

## Licence and citation

Code is provided under the MIT licence. The trajectory archives remain governed by their source CC BY 4.0 terms. Please cite the final article and both dataset DOI records when using this package.
