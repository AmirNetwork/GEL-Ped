$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$ruff = Join-Path $PSScriptRoot ".venv\Scripts\ruff.exe"

& $python -m pytest
& $ruff check .
& $python (Join-Path $PSScriptRoot "experiments\two_pedestrian_audit.py")
& $python (Join-Path $PSScriptRoot "experiments\canonical_model_validation.py")
& $python (Join-Path $PSScriptRoot "experiments\spacetime_visualization.py")
& $python (Join-Path $PSScriptRoot "experiments\timestep_sensitivity.py")
& $python (Join-Path $PSScriptRoot "experiments\empirical_dataset_summary.py")
& $python (Join-Path $PSScriptRoot "experiments\empirical_run_catalog.py")
& $python (Join-Path $PSScriptRoot "experiments\corridor_lane_formation.py")
& $python (Join-Path $PSScriptRoot "experiments\heldout_velocity_validation.py")
& $python (Join-Path $PSScriptRoot "experiments\geometry_stress_validation.py")
& $python (Join-Path $PSScriptRoot "experiments\forecast_horizon_sensitivity.py")
