# Referee-3 untouched T-junction execution record

**Author:** Amir Ghorbani

**Execution date:** 6 October 2026 (Australia/Sydney)

**Pre-outcome tag:** `gelped-v1.0-preheldout`

**Frozen analysis commit recorded by the registry:**
`4f40ec4d8544007e5fac2d30e8c88afff45764c7`

## Evidence boundary

The archive structure, task, seeds, model, guard, endpoints, equivalence
margins, and Holm family were frozen before prediction errors were computed.
The prediction program validated the official archive, exact member list,
configuration hash, graph/HGB schemas, and target-free deployment sentinel.
The resulting receipt records `target_labels_used_for_prediction=false` and
`performance_analysis_run=false`.

An initial command stopped during inventory preflight because an older
schema-only inventory referred to an earlier freeze commit. It stopped before
model fitting, held-out prediction, or performance analysis. That inventory
was preserved. The registered archive was copied byte-for-byte to a fresh
`one_shot_v1` evidence directory (same SHA-256), where the current inventory,
prediction bundles, and receipts were created without overwrite.

## Frozen inputs and receipts

- Archive bytes: `4,912,940`
- Archive SHA-256:
  `182ef75f3c0fc71497bcccf7ec5b139a02b629071ed71a022aa1aee7bd7d3827`
- Configuration SHA-256:
  `be4e1d2f6423c6a7c5f54a25754603bc1363221fc7d2d1cb3826bcfd752a7ee8`
- Registry SHA-256:
  `853b143d464ee258430990b4afbd0d0ddd4f5b5c6b5fb28cc8cc1e6d34ca6efe`
- Prediction receipt SHA-256:
  `a1253e5fa1dba36899e68b54f0502fa517a18531e60a232e4e11d0da83f5371c`
- Deployed sentinel SHA-256:
  `23dab7c011d6ff9d08bc54e60e31dac567c253d88e7bce1b4cafee2aaaf08f08`
- Verified complete runs: 11
- Verified independently seeded member fits: 25
- Primary ensemble members: frozen indices 0--4

The one-shot analysis created an exclusive
`ONE_SHOT_ANALYSIS_RECEIPT.json` before computing metrics and finalized it
with `status=complete` and hashes for all outputs. The marker prohibits rerun
or overwrite.

## Confirmatory results

Negative differences favour GEL-Ped. Complete physical runs are the
independent units.

| Endpoint | Graph | GEL-Ped | Difference | 95% run-bootstrap CI | Holm p |
|---|---:|---:|---:|---:|---:|
| RMSE (m/s) | 0.326246 | 0.314117 | -0.012129 | [-0.021062, -0.004165] | 0.0547 |
| q95 error (m/s) | 0.618924 | 0.594198 | -0.024727 | [-0.041214, -0.008462] | 0.0410 |
| CVaR95 (m/s) | 0.728099 | 0.700388 | -0.027711 | [-0.046138, -0.010460] | 0.0078 |
| Error rate above 0.5 m/s | 0.119248 | 0.104307 | -0.014941 | [-0.028100, -0.004343] | 0.0547 |

The q95 and CVaR95 improvements reject their Holm-adjusted nulls. RMSE and the
large-error rate do not cross the adjusted 0.05 threshold, although their
complete-run bootstrap intervals are below zero. GEL-Ped wins 8/11 runs for
RMSE and q95, 10/11 for CVaR95, and 7/11 (one tie) for the large-error rate.

The frozen RMSE equivalence tests do not establish equivalence because the
guarded model improves by more than either no-harm margin; this is not an
adverse finding. Superiority and equivalence are reported separately.

## Interpretation

This one-shot result supports the paper's narrow primary claim: the
instability-triggered action can reduce the upper error tail under an unseen
topology change. It does not prove universal shift robustness, calibrated
uncertainty, reciprocal collision safety, or superiority of GEL-Ped on every
run and every endpoint.
