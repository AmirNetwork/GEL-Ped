# Data provenance and expected layout

Author: Amir Ghorbani

Raw trajectories are third-party research data and are intentionally excluded
from this repository. Obtain each archive from its original provider, retain
its licence and citation files, and verify the archive before extraction. The
MIT licence in this repository applies to GEL-Ped code only.

## Data used by the current referee-3 drivers

### Jülich bidirectional corridor

- Record: [Bidirectional pedestrian flow in a corridor](https://doi.org/10.34735/ped.2013.5)
- Archive: `2013bidirectional_trajectories_txt.zip`
- SHA-256: `BDEC17C9549B1790FCC02C98A31A62D1ADBD96FA756ED540F07CE636CE97DF7E`
- Native sampling: 25 frames/s
- Extracted files: `data/raw/2013bidirectional/bi_corr_400_*.txt`

`data/splits.json` assigns seven complete runs to fitting and out-of-fold
calibration, five complete runs to familiar-corridor evaluation, and three
complete runs to the altered-layout evaluation. Frames from a physical run are
never split across those roles.

### Jülich perpendicular crossing

- Record: [Crossing, 90 degree angle](https://doi.org/10.34735/ped.2013.4)
- Archive: `trajectories_txt.zip`
- SHA-256: `64C14B11251AC6FDD530939014BF74A81CEBAE54E275465EFFA02BBBB66B3E89`
- Native sampling: 25 frames/s
- Extracted files:
  `data/raw/2013crossing90/trajectories/crossing_90_*.txt`

The current controlled-data generator includes the thirteen two-stream runs
whose names begin with `crossing_90_d_` or `crossing_90_e_`. This setting
informed method development and is not labelled as untouched confirmation.

### Registered untouched Jülich T-junction

- Record: [Jülich Data DOI 10.26165/JUELICH-DATA/YNCHYF](https://doi.org/10.26165/JUELICH-DATA/YNCHYF)
- Registered source URL:
  `http://ped.fz-juelich.de/experiments/2009.05.12_Duesseldorf_Messe_Hermes/data/zip/2009tjunction_trajectories_txt.zip`
- Archive: `2009tjunction_trajectories_txt.zip`
- Bytes: `4912940`
- SHA-256: `182EF75F3C0FC71497BCCCF7EC5B139A02B629071ED71A022AA1AEE7BD7D3827`
- Native sampling: 16 frames/s

The exact eleven member names, row counts, compressed sizes, and CRC-32 values
are frozen in `configs/referee3_heldout_registry.json`. This archive is the
one-shot topology-shift evaluation. It is deliberately stored outside
`data/raw/` and must not be opened for performance analysis until the complete
method, archive registry, endpoints, and code have been committed and tagged.
The generator validates the archive and produces target-blind predictions
before the analysis program is allowed to inspect errors.

## Additional datasets and historical exploratory workflows

These datasets are discussed in the project history and have adapters or
earlier experiment scripts in the repository, but they are **not** invoked by
`experiments/referee3_generate_predictions.py`.

### Tordeux/Jülich configurations

- Record: [Zenodo 10.5281/zenodo.1054017](https://doi.org/10.5281/zenodo.1054017)
- Expected extracted root for the earlier scripts:
  `data/raw/tordeux2017_untouched/`
- Native sampling: 16 frames/s

The archive contains unidirectional-corridor and bottleneck configurations.
Do not infer that the current referee-3 driver reruns them simply because the
loader remains available.

### ETH/UCY trajectories

The earlier five-scene audit used the ETH, HOTEL, UNIV, ZARA1, and ZARA2 text
files distributed by the
[Trajectron++ repository](https://github.com/StanfordASL/Trajectron-plus-plus)
at commit `1031c7bd1a444273af378c1ec1dcca907ba59830`. The historical scripts expect
that checkout under `tmp/Trajectron-plus-plus/experiments/pedestrians/raw/`.
The adapter is `src/pedgeom/external.py`. These files are not downloaded or
run by the current controlled-data referee-3 command sequence.

## Integrity checks

From the repository root, before extraction:

```powershell
Get-FileHash path\to\2013bidirectional_trajectories_txt.zip -Algorithm SHA256
Get-FileHash path\to\trajectories_txt.zip -Algorithm SHA256
Get-FileHash path\to\2009tjunction_trajectories_txt.zip -Algorithm SHA256
```

For a schema-only T-junction check that does not fit models or inspect outcome
errors:

```powershell
.\.venv\Scripts\python experiments\referee3_generate_heldout.py `
  --archive path\to\2009tjunction_trajectories_txt.zip `
  --work-directory path\to\scratch `
  --inventory-only
```

Processed files in `data/processed/` are generated artifacts, not substitute
copies of the raw datasets. Large referee-3 caches and prediction bundles are
ignored by Git. The scientific specification is `configs/referee3_lock.json`;
the one-shot archive specification is
`configs/referee3_heldout_registry.json`.
