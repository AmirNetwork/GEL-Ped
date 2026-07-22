# Pedestrian Geometry Research Workspace

This repository rebuilds and tests the method in *Spacetime metric for pedestrian
movement* (arXiv:2211.10792), then develops an empirically testable geometric model.

The project deliberately separates three objects:

1. `legacy_reset`: the paper's stated zero-velocity-at-each-step update;
2. `corrected_ode`: persistent-velocity integration of the corrected Newtonian equations;
3. `anisotropic_pilot`: a calibration-ready geometric navigation model.

The original paper is treated as a scientific baseline, not silently modified. Every
correction is documented in `reports/mathematical_audit.md`.

## Layout

- `src/pedgeom/`: model and numerical integration code
- `tests/`: dimensional, analytic, and numerical regression tests
- `experiments/`: reproducible experiment entry points
- `reports/`: audit and experiment notes
- `data/`: raw and processed empirical data (raw data are not committed)
- `figures/`: generated figures

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

Run the initial two-pedestrian comparison:

```powershell
.venv\Scripts\python experiments\two_pedestrian_audit.py
```

Run all current checks and experiments:

```powershell
.\run_research.ps1
```

The empirical corridor archive is not bundled in git. Its source, licence, checksum,
and expected location are recorded in `data/README.md`.

## Research rule

No publication claim is made unless it survives held-out empirical validation against
calibrated mechanistic baselines with sensitivity and timestep-convergence analyses.

The first anisotropic pilot is retained as a negative result: it does not yet satisfy
the joint collision-safety, speed, and lane-formation criteria.

