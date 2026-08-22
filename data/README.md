# Data provenance and expected layout

Author: Amir Ghorbani

The raw trajectory archives are excluded from the code package because they are large.
Both sources are public under CC BY 4.0. Download them from the DOI landing pages and
verify their checksums before running the experiments.

## Bidirectional corridor (development and internal validation)

- Dataset: *Bidirectional pedestrian flow in a corridor*
- DOI: https://doi.org/10.34735/ped.2013.5
- Sampling rate: 25 frames/s
- Archive name: `2013bidirectional_trajectories_txt.zip`
- SHA-256: `BDEC17C9549B1790FCC02C98A31A62D1ADBD96FA756ED540F07CE636CE97DF7E`
- Expected extracted files: `data/raw/2013bidirectional/bi_corr_400_*.txt`

The loader reads pedestrian ID, frame, x, y, and height in centimetres and converts
coordinates to SI units. `data/splits.json` assigns seven complete runs to calibration,
five complete runs to held-out validation, and three geometry variants to stress testing.

## Perpendicular crossing (external validation only)

- Dataset: *Crossing, 90 degree angle*
- DOI: https://doi.org/10.34735/ped.2013.4
- Sampling rate: 25 frames/s
- Archive name: `trajectories_txt.zip`
- SHA-256: `64C14B11251AC6FDD530939014BF74A81CEBAE54E275465EFFA02BBBB66B3E89`
- Expected extracted files: `data/raw/2013crossing90/trajectories/crossing_90_*.txt`

Only the thirteen two-stream runs whose names begin with `crossing_90_d_` or
`crossing_90_e_` are used. No crossing run is used for feature design, coefficient
fitting, range selection, or model selection.

## Integrity check

From the repository root:

```powershell
Get-FileHash data\raw\2013bidirectional_trajectories_txt.zip -Algorithm SHA256
Get-FileHash data\raw\2013crossing90\trajectories_txt.zip -Algorithm SHA256
```

The processed CSV and JSON files in `data/processed/` are generated outputs, not an
alternative copy of the raw trajectories.
