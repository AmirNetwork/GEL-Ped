# Data provenance and expected layout

Author: Amir Ghorbani

Raw third-party trajectories are excluded from the code package. Download them
from the cited archives, retain their original terms, and verify checksums before
running the experiments.

## Juelich bidirectional corridor

- Dataset: *Bidirectional pedestrian flow in a corridor*
- DOI: https://doi.org/10.34735/ped.2013.5
- Sampling rate: 25 frames/s
- Archive name: `2013bidirectional_trajectories_txt.zip`
- SHA-256: `BDEC17C9549B1790FCC02C98A31A62D1ADBD96FA756ED540F07CE636CE97DF7E`
- Expected files: `data/raw/2013bidirectional/bi_corr_400_*.txt`

The loader converts centimetres to SI units. `data/splits.json` assigns seven
complete runs to calibration, five to familiar-flow testing, and three changed
length/exit arrangements to the altered-geometry test.

## Juelich perpendicular crossing

- Dataset: *Crossing, 90 degree angle*
- DOI: https://doi.org/10.34735/ped.2013.4
- Sampling rate: 25 frames/s
- Archive name: `trajectories_txt.zip`
- SHA-256: `64C14B11251AC6FDD530939014BF74A81CEBAE54E275465EFFA02BBBB66B3E89`
- Expected files:
  `data/raw/2013crossing90/trajectories/crossing_90_*.txt`

The thirteen two-stream runs beginning `crossing_90_d_` or `crossing_90_e_`
are used. This archive influenced method exploration and is treated as a
development-informed stress test, not a clean confirmatory or multi-site test.

## Additional Tordeux configurations

- Archive DOI: https://doi.org/10.5281/zenodo.1054017
- Expected root: `data/raw/tordeux2017_untouched/`
- Sampling rate used by the original files: 16 frames/s
- Configurations used here: eight unidirectional-corridor runs and four
  bottleneck runs

These complete-run evaluations are reported as additional descriptive tests.
The exact file catalogue and sample counts are written to the referee2 output
tables.

## ETH/UCY pedestrian trajectories

The five-scene external audit uses ETH, HOTEL, UNIV, ZARA1, and ZARA2 files
distributed by the Trajectron++ repository at commit
`1031c7bd1a444273af378c1ec1dcca907ba59830`. The official leave-one-scene-out
protocol trains on the other scenes, calibrates the guard on their supplied
validation split, and evaluates the held scene once. At 2.5 Hz, two intervals
provide the 0.8-s history and one interval provides the 0.4-s target.

## Integrity checks

From the repository root:

```powershell
Get-FileHash data\raw\2013bidirectional_trajectories_txt.zip -Algorithm SHA256
Get-FileHash data\raw\2013crossing90\trajectories_txt.zip -Algorithm SHA256
```

Processed CSV and JSON files in `data/processed/` are generated outputs, not
alternative copies of the raw trajectories. The final specification and honest
development chronology are recorded in `configs/referee2_lock.json`.
