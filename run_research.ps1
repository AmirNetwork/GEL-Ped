# Author: Amir Ghorbani
# Reproduce the GEL-Ped analyses, figures, tests, and integrated manuscript.
$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$ruff = Join-Path $PSScriptRoot ".venv\Scripts\ruff.exe"

& $python -m pytest
& $ruff check .
& $python (Join-Path $PSScriptRoot "experiments\major_revision_analysis.py")
& $python (Join-Path $PSScriptRoot "experiments\capacity_matched_neural_ensemble.py")
& $python (Join-Path $PSScriptRoot "experiments\neural_horizon_sensitivity.py")
& $python (Join-Path $PSScriptRoot "experiments\refresh_confirmatory_holm.py")
& $python (Join-Path $PSScriptRoot "experiments\reviewer_revision_analysis.py")
& $python (Join-Path $PSScriptRoot "experiments\shift_routed_upgrade.py")
& $python (Join-Path $PSScriptRoot "experiments\shift_routed_seed_sensitivity.py")
& $python (Join-Path $PSScriptRoot "experiments\shift_routed_wall_check.py")
& $python (Join-Path $PSScriptRoot "experiments\autoregressive_rollout.py")
& $python (Join-Path $PSScriptRoot "experiments\rollout_statistics.py")
& $python (Join-Path $PSScriptRoot "experiments\shift_routed_data_efficiency.py")
& $python (Join-Path $PSScriptRoot "experiments\final_publication_figures.py")
& $python (Join-Path $PSScriptRoot "experiments\manuscript_consistency_audit.py")
