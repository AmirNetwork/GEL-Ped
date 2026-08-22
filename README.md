# GEL-Ped

**GEL-Ped** stands for **Geometry-Encoded Learning for Pedestrian Forecasting**. This repository, authored by **Amir Ghorbani**, reproduces the experiments and integrated manuscript for *GEL-Ped: Geometry-encoded learning for pedestrian forecasting under limited data and unseen flow topology*.

The method combines two experts:

1. a direct invariant neural predictor for flexible interpolation; and
2. a neural residual around a route-aligned tensor prior, attenuated outside calibration support.

The reported GEL-Ped model is their calibration-selected convex blend. Its machine-readable model identifier is `gel_ped`. The tensor representation is a behavioural geometry and visualization device; it is not a claim that relativistic effects govern walking.

## Practical result

At a 0.4 s forecast horizon, all model choices are made from seven complete corridor calibration runs and then frozen. On untouched runs:

- with all seven calibration runs, the direct network is slightly better in familiar corridors (0.1819 versus 0.1838 m/s RMSE), so it remains the sensible choice for abundant same-topology data;
- on thirteen external 90-degree crossing runs, GEL-Ped improves the direct network by 1.38% (0.3588 versus 0.3638 m/s) and a protocol-matched 2024 goal-stable hybrid by 1.49%; both remain significant after confirmatory Holm correction;
- across 0.2, 0.4, 0.8, and 1.2 s horizons, crossing reductions from the direct network are 3.55%, 1.38%, 4.56%, and 2.94%, with significance retained after correction across horizons;
- after architecture selection is fixed, GEL-Ped lowers pooled error for all 127 possible subsets of one to seven fitting runs, including a 14.12% mean reduction for singleton choices;
- across all twenty-one untouched runs, it reduces error by 13.28% relative to calibrated Social Force and by 6.53% relative to an unrestricted tensor-feature linear model.

The intended decision rule is therefore explicit: use the direct neural expert for well-sampled familiar-scene interpolation; use GEL-Ped when calibration data are scarce or the deployment topology may change.

## Manuscript-to-code map

The implementation follows the numbered equations in the integrated manuscript:

- Eqs. (1)-(5): forecast samples and observed interaction fields in `pedgeom.calibration.build_velocity_samples`.
- Eqs. (6)-(7): route-aligned tensor response in `pedgeom.calibration.TensorGeometryModel`.
- Eqs. (8)-(9): support score, gate, and structured residual in `pedgeom.benchmarks.TensorResidualRegressor`.
- Eq. (10): matched direct neural expert in `pedgeom.benchmarks.InvariantRegressor` and `fit_interaction_mlp`.
- Eq. (11): complete GEL-Ped blend in `pedgeom.benchmarks.GELPedRegressor`.
- Eq. (12): calibrated Social Force reference in `pedgeom.calibration.SocialForceResponseModel`.
- Eq. (13): run-balanced primary metric in `pedgeom.calibration.velocity_metrics`.

## Repository map

- `src/pedgeom/`: fields, tensor model, neural benchmarks, calibration, metrics, and statistics
- `experiments/major_revision_analysis.py`: frozen primary analysis
- `experiments/data_efficiency_analysis.py`: calibration-data sensitivity
- `experiments/neural_horizon_sensitivity.py`: frozen cross-horizon neural transfer
- `experiments/refresh_confirmatory_holm.py`: confirmatory multiplicity update
- `experiments/publication_figures.py`: experiment-context and real-trajectory visuals
- `experiments/publication_upgrade_figures.py`: architecture and result visuals
- `experiments/manuscript_consistency_audit.py`: data, benchmark-fairness, and manuscript cross-check
- `configs/major_revision.json`: machine-readable experiment configuration
- `tests/`: analytic, numerical, data, statistical, and regression tests
- `data/splits.json`: immutable complete-run partition
- `data/processed/`: run-level results, uncertainty, diagnostics, and fitted values
- `figures/`: generated publication figures
- `manuscript/`: Markdown source, LaTeX builder, bibliography, and Overleaf project

## Reproduce

Python 3.11 or newer is required. Download the two CC BY 4.0 trajectory archives described in `data/README.md`, place them in the expected directories, and run:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.\run_research.ps1
```

The runner executes the tests and lint checks, rebuilds all primary and sensitivity results, regenerates the figures, and writes the single-file Overleaf manuscript with its integrated appendices. Compile `manuscript/overleaf_trc/main.tex` with pdfLaTeX, BibTeX, and two further pdfLaTeX passes.

The frozen stochastic seed is `20260722`. Raw trajectory archives are not redistributed; processed result summaries contain no substitute copy of the source trajectories.

## Licence and citation

Code is provided under the MIT licence. The trajectory archives remain governed by their source CC BY 4.0 terms. Please cite the final article and both dataset DOI records when using this package.
