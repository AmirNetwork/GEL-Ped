# Referee-3 untouched T-junction archive inventory

This record was made after the analysis freeze at commit
`0825df3c5ded408f0f2165b3ba2db7d2149a5b0e` and before any model-error or
performance calculation on the T-junction data.

- Official URL: `http://ped.fz-juelich.de/experiments/2009.05.12_Duesseldorf_Messe_Hermes/data/zip/2009tjunction_trajectories_txt.zip`
- Downloaded archive: `E:\GEL-Ped-TRC-Review-Compute\tjunction\2009tjunction_trajectories_txt.zip`
- Size: 4,912,940 bytes
- SHA-256: `182ef75f3c0fc71497bcccf7ec5b139a02b629071ed71a022aa1aee7bd7d3827`
- Schema-only inventory JSON: `E:\GEL-Ped-TRC-Review-Compute\tjunction\archive_inventory.json`
  (SHA-256 `4ce8f655e8467cab617178daeed2b5fa4cadbd3fb9ab5bdd229df7ae7d39f701`)
- Structure: 11 flat text members, no header/comment rows, five whitespace-separated numeric columns in every data row
- Frozen schema: pedestrian identifier, frame, x (cm), y (cm), z/height (cm)
- Re-frozen timing before execution: source 16 Hz; a past-only two-sample
  linear projection to the common 10 Hz grid. The two source observations and
  the age of the latest one must each fit within 1.5 native intervals
  (0.09375 s), so missing-frame gaps are not bridged. Origins without two
  valid prior observations are dropped, and no state is projected beyond a
  trajectory's observed support.

| Member | Rows | Bytes | Compressed bytes | CRC-32 |
|---|---:|---:|---:|---:|
| `ko-240-050-240.txt` | 13,569 | 450,145 | 144,922 | `32aadbe3` |
| `ko-240-060-240.txt` | 15,331 | 507,997 | 163,253 | `f344b5ae` |
| `ko-240-080-240.txt` | 32,058 | 1,081,280 | 339,785 | `6a6e4cc9` |
| `ko-240-100-240.txt` | 33,645 | 1,133,825 | 355,704 | `1c4b38e1` |
| `ko-240-120-240.txt` | 67,992 | 2,321,821 | 715,374 | `19697431` |
| `ko-240-150-240.txt` | 75,373 | 2,569,350 | 791,924 | `95dba8eb` |
| `ko-240-240-240.txt` | 80,721 | 2,769,576 | 845,562 | `9826fd66` |
| `ko-300-050-300.txt` | 13,749 | 458,499 | 146,927 | `2859456c` |
| `ko-300-080-300.txt` | 23,378 | 784,613 | 248,593 | `b9cc7f04` |
| `ko-300-120-300.txt` | 51,078 | 1,731,663 | 538,643 | `fc390c8f` |
| `ko-300-150-300.txt` | 58,968 | 2,001,532 | 620,119 | `6a30b0ef` |

Only archive names, byte metadata, row/column structure, and numeric parseability
were inspected. No forecast error, baseline comparison, gate outcome, or tuning
statistic was computed.
