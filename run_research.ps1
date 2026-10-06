# Author: Amir Ghorbani
# Reproduce the controlled-data referee-3 pipeline. This script never opens
# the registered one-shot T-junction outcomes.
[CmdletBinding()]
param(
    [ValidateRange(1, 64)]
    [int]$Workers = 4,
    [switch]$SkipQualityChecks,
    [switch]$SkipTransformer,
    [switch]$SkipPlanner
)

$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing .venv. Follow the installation steps in README.md first."
}

# Avoid nested BLAS thread pools competing with process-level member training.
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:BLIS_NUM_THREADS = "1"
$env:VECLIB_MAXIMUM_THREADS = "1"

Push-Location $PSScriptRoot
try {
    if (-not $SkipQualityChecks) {
        & $python -m pytest
        & $python -m ruff check .
    }

    & $python experiments\referee3_generate_predictions.py --workers $Workers
    & $python experiments\referee3_analyse.py

    if (-not $SkipTransformer) {
        & $python experiments\referee3_transformer_generalization.py
    }
    if (-not $SkipPlanner) {
        & $python experiments\referee3_reactive_planner.py
    }

    Write-Host "Controlled-data referee-3 pipeline completed."
    Write-Warning (
        "The untouched T-junction was not run. Follow the frozen, explicit " +
        "one-shot procedure in REPRODUCIBILITY.md only after tagging the protocol."
    )
}
finally {
    Pop-Location
}
