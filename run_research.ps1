# Author: Amir Ghorbani
# Reproduce the GEL-Ped analyses, figures, tests, and integrated manuscript.
$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$ruff = Join-Path $PSScriptRoot ".venv\Scripts\ruff.exe"

& $python -m pytest
& $ruff check .
& $python (Join-Path $PSScriptRoot "experiments\major_revision_analysis.py")
& $python (Join-Path $PSScriptRoot "experiments\capacity_matched_neural_ensemble.py")
& $python (Join-Path $PSScriptRoot "experiments\data_efficiency_analysis.py")
& $python (Join-Path $PSScriptRoot "experiments\neural_horizon_sensitivity.py")
& $python (Join-Path $PSScriptRoot "experiments\refresh_confirmatory_holm.py")
& $python (Join-Path $PSScriptRoot "experiments\publication_figures.py")
& $python (Join-Path $PSScriptRoot "experiments\publication_upgrade_figures.py")
& $python (Join-Path $PSScriptRoot "experiments\manuscript_consistency_audit.py")
