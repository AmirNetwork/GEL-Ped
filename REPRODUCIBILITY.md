# GEL-Ped reproducibility guide

Author: Amir Ghorbani

This guide describes the referee-3 code path for *GEL-Ped: An
instability-triggered safeguard for short-horizon pedestrian prediction under
changing crowd flows*. It separates repeatable development analyses from the
registered one-shot T-junction evaluation.

## 1. Scientific identity

Record these values with every run:

```powershell
git rev-parse HEAD
git status --short
Get-FileHash configs\referee3_lock.json -Algorithm SHA256
Get-FileHash configs\referee3_heldout_registry.json -Algorithm SHA256
```

`configs/referee3_lock.json` fixes the task, inputs, experts, guard,
replication design, endpoints, controls, planner, and multiplicity family.
`configs/referee3_heldout_registry.json` binds the untouched archive and its
members to a frozen analysis commit. Generated manifests additionally record
the hashes of relevant code, inputs, fitted-member caches, and prediction
bundles.

Never modify a frozen JSON file to make an observed result more favourable.
Any scientifically justified post-freeze change requires a new namespace,
commit, tag, and an explicit exploratory label.

## 2. Reference environment

The reference run uses Python 3.12 and the exact versions in
`requirements-lock.txt` and `requirements-graph.txt`. Python 3.11 or newer is
supported.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
.\.venv\Scripts\python -m pip install -r requirements-graph.txt
.\.venv\Scripts\python -m pip install -e . --no-deps
.\.venv\Scripts\python -m pip check
```

Before a multiprocessing run, limit numerical libraries to one thread per
worker:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:BLIS_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"
```

Member seeds, ensemble draws, bootstrap seeds, and permutation seeds are fixed
in the configuration. Exact floating-point equality across different PyTorch,
BLAS, CPU, and GPU builds is not promised. The held-out path therefore also
checks a frozen, target-free deployed sentinel before accepting a refit.

## 3. Inputs and data roles

Place the bidirectional-corridor and crossing files exactly as documented in
`data/README.md`. Verify the source archives before extraction. The current
controlled-data referee-3 generator reads:

- `data/splits.json`;
- `data/raw/2013bidirectional/*.txt`;
- `data/raw/2013crossing90/trajectories/*.txt`;
- `data/processed/prospective_selection.json` for the previously selected
  fallback settings; and
- `configs/referee3_lock.json`.

The current `referee3_*` controlled-data driver does not automatically run the
older Tordeux or ETH/UCY workflows. Their adapters and historical scripts are
retained for provenance; they must not be described as outputs of
`run_research.ps1`.

Large caches can exceed the free space of a system drive. It is safe to make
`data/processed/referee3_member_cache` and
`data/processed/referee3_predictions` directory junctions to a larger local
disk before the first run. Keep the logical repository paths unchanged because
the manifests store repository-relative names.

## 4. Quality checks

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
git diff --check
```

The tests include causal history construction, past-only 16-to-10-Hz
resampling, target-blind inference, graph and HGB feature schemas, time-window
reset rules, tail statistics, held-out receipt validation, and planner replay.

## 5. Repeatable controlled-data pipeline

Run the following commands in order from the repository root:

```powershell
.\.venv\Scripts\python experiments\referee3_generate_predictions.py --workers 4
.\.venv\Scripts\python experiments\referee3_analyse.py
.\.venv\Scripts\python experiments\referee3_transformer_generalization.py
.\.venv\Scripts\python experiments\referee3_reactive_planner.py
```

The first command writes resumable member caches, target-blind prediction
bundles, a deployed sentinel, and `referee3_training_manifest.json`. The second
uses those fixed predictions for ensemble-draw, tail, detection, targeting,
decomposition, delayed-expert, and equivalence analyses. The compact
Transformer check has its own cache namespace. The planner audit requires
`referee3_primary_calibration.json`, so it must follow the main analysis.

`run_research.ps1` executes the same sequence and supports `-Workers`,
`-SkipQualityChecks`, `-SkipTransformer`, and `-SkipPlanner`. It never invokes
the untouched T-junction programs.

Do not commit the large cache and prediction directories. Preserve the
manifests, compact tables, frozen configuration, and exact release tag needed
to audit them.

## 6. Untouched T-junction firewall

The T-junction is not another tuning set. Before the first labelled outcome is
examined:

1. Complete and test all method and analysis code.
2. Commit the clean protocol and create a signed or annotated pre-held-out tag.
3. Confirm that the registry points to the intended freeze and that its
   configuration hash is unchanged.
4. Download the registered archive from the original provider and verify its
   byte count and SHA-256.
5. Run only the inventory mode first; it parses archive metadata but computes
   no performance measure.
6. Generate target-blind prediction bundles exactly once. Archive the receipt,
   caches, bundles, commit ID, environment, and console log before analysis.
7. Run the frozen confirmatory analysis once, preserve its output directory,
   and report every endpoint, including null or adverse outcomes.
8. Do not change the model, threshold, cap, fallback, inclusion rule,
   endpoint, or multiplicity family in response to the held-out result.

Schema-only inventory:

```powershell
.\.venv\Scripts\python experiments\referee3_generate_heldout.py `
  --archive path\to\2009tjunction_trajectories_txt.zip `
  --work-directory path\to\scratch `
  --inventory-only
```

Irreversible prediction step, after the freeze checks above:

```powershell
.\.venv\Scripts\python experiments\referee3_generate_heldout.py `
  --archive path\to\2009tjunction_trajectories_txt.zip `
  --work-directory path\to\scratch `
  --sentinel data\processed\referee3_deployed_sentinel.npz `
  --workers 4 `
  --execute-once
```

Frozen analysis, only after prediction artifacts have been archived:

```powershell
.\.venv\Scripts\python experiments\referee3_analyse_heldout.py `
  --prediction-directory path\to\scratch\referee3_tjunction_predictions `
  --output-directory path\to\immutable-heldout-report
```

Use a new, empty output directory. Treat both prediction and analysis
directories as immutable evidence after completion.

## 7. Manuscript-to-code map

| Manuscript quantity | Implementation | Verification |
|---|---|---|
| 0.8-s velocity anchor and 0.4-s target (Eq. 1) | `build_velocity_samples` in `src/pedgeom/calibration.py` | `tests/test_sample_construction.py`, `tests/test_resampling.py` |
| causal interaction fields | `_interaction_features` in `src/pedgeom/calibration.py` | `tests/test_calibration.py`, `tests/test_referee3_graph_schema.py` |
| graph-member correction (Eq. 2) | `_graph_arrays`, `_InteractionGraph`, and `fit_graph_interaction_network` in `src/pedgeom/graph_baseline.py` | `tests/test_referee3_graph_schema.py`, `tests/test_referee3_prediction_pipeline.py` |
| run-balanced graph loss | `fit_graph_interaction_network` in `src/pedgeom/graph_baseline.py` | graph and prediction-pipeline tests |
| feature-based HGB fallback (Eq. 4) | `fit_gradient_boosted_residual` in `src/pedgeom/benchmarks.py` | `tests/test_benchmarks.py`, `tests/test_referee3_hgb_schema.py` |
| local member spread (Eq. 5) | member predictions in `experiments/referee3_generate_predictions.py`; score construction in `experiments/referee3_analyse.py` | `tests/test_referee3_analysis.py` |
| trailing 1.6-s causal mean | `causal_trailing_time_mean` in `src/pedgeom/runtime_guard.py` | `tests/test_runtime_guard.py` |
| finite-sample calibration quantile | `finite_sample_upper_quantile` in `src/pedgeom/runtime_guard.py` | `tests/test_runtime_guard.py` |
| one-sided exponential graph weight and blend (Eq. 6) | `one_sided_gate_weights` in `src/pedgeom/runtime_guard.py`; `selected_predictions`/analysis assembly in `experiments/referee3_analyse.py` | `tests/test_runtime_guard.py`, `tests/test_referee3_analysis.py` |
| tail, detection, retention, and decomposition endpoints | `src/pedgeom/runtime_guard.py` and `experiments/referee3_analyse.py` | runtime-guard and analysis tests |
| delayed-error expert comparator | `causal_delayed_two_expert_weights` in `src/pedgeom/runtime_guard.py` | runtime-guard and analysis tests |
| reactive consequence audit | `src/pedgeom/reactive_planner.py` and `experiments/referee3_reactive_planner.py` | `tests/test_reactive_planner.py` |

## 8. Release checks

Before publishing a code or manuscript archive:

```powershell
git status --short
git diff --check
.\.venv\Scripts\python -m pip check
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
Get-FileHash manuscript\overleaf_trc\main.pdf -Algorithm SHA256
```

The release archive should contain source, tests, configuration, compact
result tables required by the manuscript, `CITATION.cff`, `LICENSE`, and the
Overleaf package. It should not contain raw third-party trajectories, local
virtual environments, machine-specific caches, temporary renders, secrets, or
absolute workstation paths.
