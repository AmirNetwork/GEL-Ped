# GEL-Ped

**GEL-Ped (Guarded Ensemble Learning for Pedestrians)** is an
instability-triggered safeguard for short-horizon pedestrian prediction under
changing crowd flows. It predicts mean planar velocity over the next 0.4 s
from a causal 0.8-s observation history.

GEL-Ped is authored by **Amir Ghorbani**. The accompanying manuscript is titled
*GEL-Ped: An instability-triggered safeguard for short-horizon pedestrian
prediction under changing crowd flows*.

## Method in one paragraph

Five independently initialized graph predictors form the default forecast.
Their pedestrian-local disagreement is averaged over a trailing 1.6-s causal
window and compared with an envelope estimated by leave-one-complete-run-out
calibration. The graph ensemble keeps weight one while disagreement remains
inside that envelope. Only an exceedance moves the forecast smoothly toward a
feature-based histogram-gradient-boosting fallback with an explicit output
speed cap. The method returns both the forecast and the graph weight, so an
operational system can log or act on an intervention. Its scope is deliberately
narrow: it responds to detectable member instability and is not a universal
distribution-shift detector, uncertainty guarantee, or safety certificate.

The immutable scientific specification is
[`configs/referee3_lock.json`](configs/referee3_lock.json). The untouched
T-junction registry is
[`configs/referee3_heldout_registry.json`](configs/referee3_heldout_registry.json).

## Repository map

- `src/pedgeom/calibration.py`: causal samples, interaction fields, and metrics
- `src/pedgeom/graph_baseline.py`: anchored graph interaction predictor
- `src/pedgeom/benchmarks.py`: feature-based HGB fallback and matched experts
- `src/pedgeom/runtime_guard.py`: time-windowed score, one-sided gate, tail and
  online-control utilities
- `src/pedgeom/reactive_planner.py`: paired fixed/adaptive open-loop planner
  replay
- `experiments/referee3_generate_predictions.py`: leakage-resistant
  out-of-fold and controlled-data prediction bundles
- `experiments/referee3_analyse.py`: frozen draw, tail, targeting,
  decomposition, online, and equivalence analyses
- `experiments/referee3_transformer_generalization.py`: matched compact
  Transformer-backbone check on the controlled-data batches
- `experiments/referee3_reactive_planner.py`: consequence and latency audit on
  those controlled-data bundles
- `experiments/referee3_generate_heldout.py`: prediction-only, execute-once
  untouched T-junction path
- `experiments/referee3_analyse_heldout.py`: frozen confirmatory held-out
  analysis
- `manuscript/overleaf_trc/`: manuscript source, bibliography, and figures
- `tests/`: causality, leakage, schema, statistics, and planner regression tests

The current `referee3_*` automation covers the Jülich corridor/crossing data
and the separately registered T-junction. Tordeux and ETH/UCY adapters and
earlier exploratory workflows remain available in the repository, but they
are not silently included by the referee-3 driver. This distinction prevents a
single command from being misrepresented as reproducing every historical
analysis.

The equation-to-code correspondence and exact execution order are documented
in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Installation

Python 3.11 or newer is required. The reference environment was run with
Python 3.12 on CPU.

For a convenient editable development environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pip install -r requirements-graph.txt
```

For the exact recorded package versions, install the two lock files first and
then install this package without resolving dependencies again:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python -m pip install -r requirements-graph.txt
.\.venv\Scripts\python -m pip install -e . --no-deps
```

`requirements-graph.txt` selects the official CPU PyTorch wheel index. A CUDA
build is optional and is not required to reproduce the reported protocol.

## Data

Raw third-party trajectories are not redistributed. Sources, checksums,
expected directories, and dataset roles are listed in
[`data/README.md`](data/README.md). Dataset copyrights and licence terms remain
with their respective providers.

The untouched T-junction evaluation is different from the development runs.
Do **not** execute it while changing features, thresholds, models, endpoints,
or inclusion rules. Freeze and tag the complete protocol first, verify the
registered archive and sentinel, generate target-blind predictions once, and
only then run the locked confirmatory analysis. See the explicit firewall in
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Reproduce the controlled-data development analysis

After placing the controlled archives in the documented locations:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\ruff check .
.\.venv\Scripts\python experiments\referee3_generate_predictions.py --workers 4
.\.venv\Scripts\python experiments\referee3_analyse.py
.\.venv\Scripts\python experiments\referee3_transformer_generalization.py
.\.venv\Scripts\python experiments\referee3_reactive_planner.py
```

`run_research.ps1` runs this controlled-data development sequence but
intentionally never opens the untouched T-junction outcomes. The training
scripts are resumable; their large member caches and prediction bundles are
generated locally and excluded from Git.

## Licence and citation

The source code is released under the [MIT License](LICENSE). This licence does
not cover third-party datasets, photographs, or publisher-formatted material.
Please cite the accompanying article and the software metadata in
[`CITATION.cff`](CITATION.cff), as well as every original dataset record used
in an analysis.
