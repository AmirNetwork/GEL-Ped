# Referee-3 manuscript revision map

This map links each new analysis to its manuscript destination. It contains no numerical result and must not be used to infer one. Final prose must follow the signed artifact, including null or adverse findings.

## Proposed paper spine

| Manuscript part | Purpose after revision | Required source |
|---|---|---|
| Title and abstract | Present GEL-Ped as an instability-triggered safeguard for short-horizon pedestrian prediction; one final sentence states the verified scope finding | `reports/referee3_intro_related_draft.tex`; final referee-3 result manifest |
| Introduction | Motivate the fast response task and state four questions: detection, action, scope, use | `reports/referee3_intro_related_draft.tex` |
| Related Work | Connect pedestrian/hybrid prediction, shift monitoring, selective/adaptive methods, and runtime assurance; end with positioning table | `reports/literature_verification.md`; `reports/referee3_intro_related_draft.tex` |
| Method | Define causal features, common expert pair, time-indexed instability score, finite-sample envelope, and deployed/cross-fitted model roles | `configs/referee3_lock.json`; prediction manifest; A1-A3 audits |
| Experimental design | Freeze data roles, endpoints, equivalence margins, controls, ensemble draws, held-out rule, and planner policy before results | `configs/referee3_lock.json`; `reports/revision_decision_log.md` |
| Results | Organise by RQ1 detection, RQ2 action, RQ3 scope, RQ4 use | Artifacts listed below |
| Discussion | State what disagreement can detect, what it cannot detect, and what an intervention should trigger | decomposition, held-out, planner, and latency artifacts |
| Conclusion | One bounded value proposition; no universal shift, safety, or uncertainty-coverage claim | final verified results only |

## Analysis-to-section map

| Analysis | Manuscript destination | Minimum presentation | Claim rule |
|---|---|---|---|
| Tail quantiles, CVaR, large-error rate | Experimental endpoints; Results RQ1/RQ2 | Run-macro table for graph, GEL-Ped, HGB, fixed blend, and oracle; q90/q95/q99, CVaR95/CVaR99, error-rate threshold definition | Call tail control supported only if benefit is stable across runs/draws and not created by fixed blending |
| Detection AUROC/AUPRC and risk-retention | Results RQ1 | Separate detection figure; compare spread, Mahalanobis, graph-HGB disagreement, speed, and density proxy; report AURC definition/direction | If spread is not competitive, describe it as a simple monitor rather than a superior detector |
| Fixed blend, realised-mean blend, within-run permuted gate | Results RQ2 | Targeting table and high-instability strata figure with graph/HGB/oracle | Use “targeted” only if GEL-Ped beats matched fixed and permuted controls in the relevant tail; otherwise use “adaptive blending” |
| Matched gate controls | Methods and Results RQ1/RQ2 | One common graph/HGB expert-pair description; gate-only comparison table with firing rate | No gate ranking may mix fallback types or tuning budgets |
| Shared-error/member-variance decomposition | Results RQ3; opening of Discussion | By regime and instability stratum; use “shared ensemble error,” not population bias unless formally estimated | Scope the method to variance-type instability; quantify but do not hide the shared-error component |
| Independent member pool and ensemble-size sweep | Experimental design; Results RQ3 | Draw distribution, divergent-member frequency, M sensitivity, run/draw uncertainty | Remove generality/significance language if the effect is seed-specific |
| In-support equivalence | Results RQ2 | TOST/confidence intervals at locked margin; guard firing rate beside accuracy | Never infer equivalence from non-significance |
| Untouched T-junction shift | Data-role table; Results RQ3 | One-shot execution record, confirmatory family, Holm adjustment, grouped inference | Report once without tuning. If neutral, present a scope/null result; if adverse beyond margin, stop the positive submission claim |
| Delayed-error online expert baseline | Baselines; Results RQ2 | Overall and cold-start comparison, calibration-selected learning/share rates | Explain whether spread offers information before delayed labels or adds value beyond online loss weighting |
| Second transformer-style backbone and training recipe | Methods appendix; Results RQ3 | Per-run mean/tail results, firing rate, manifest; early stopping/standardisation/weight averaging | Claim architecture transfer only if supported; otherwise restrict GEL-Ped to the tested graph setting |
| Time-window, quantile, and gate-shape sensitivities | Method; calibration appendix | Realised timing audit, run-bootstrap threshold/scale, firing rates, window/shape table, calibration distribution with g(E) | Primary specification remains frozen; sensitivity cannot choose a new held-out winner |
| Reactive planner and latency | Results RQ4 | Fixed versus adaptive paired replay: intrusions, separation, travel time, stops, failures; CPU latency at stated crowd size | Call it replayed operational evidence, not closed-loop safety proof; omit operational-benefit claim if not completed |
| Redesigned separation/TTC audit | Results RQ4 or appendix | Defined event unit/base rate; continuous separation/TTC errors by intervention stratum | Retire the former high-prevalence screen from the main claim |
| Failed uncertainty coverage | Limitations only | One concise negative statement and adaptive-conformal citation | No calibrated-region or coverage-guarantee language unless a separately frozen study succeeds |

## Required tables and figures

| Item | Content | Artifact dependency |
|---|---|---|
| Table 1: data roles | Dataset/run family, sampling rate, task timing, role, development visibility, model fit/calibration/test use | lock file, prediction manifest, untouched execution record |
| Table 2: primary mean/tail evidence | Run-macro mean plus tail family and draw spread | `referee3_primary_run_metrics.csv`, `referee3_draw_run_metrics.csv` |
| Table 3: matched controls | Common expert pair; guard signal/rule, firing rate, mean and tail endpoints | primary metrics, detection metrics, firing rates |
| Table 4: scope/replication | Shared error versus member variance; ensemble-size/draw robustness; second backbone | decomposition, draw, transformer artifacts |
| Figure 1: chronology | Dataset timeline and role, without review-history language | data-role registry |
| Figure 2: method | Graph members, time-indexed spread, cross-fitting loop, envelope, HGB fallback, intervention output | locked method |
| Figure 3: detection | AUROC/AUPRC and risk-retention; uncluttered panels | detection/risk-retention artifacts |
| Figure 4: targeting/tail | Error strata with graph, GEL-Ped, HGB, fixed, permuted, oracle | targeting artifacts |
| Figure 5: scope | Shared-error/member-variance decomposition plus draw/size robustness | decomposition/draw artifacts |
| Figure 6: operational use | Planner safety-efficiency and latency, only if completed | planner/latency artifacts |

## Conditional result language

- **Positive and replicated:** “The guard reduced [specified endpoint] across [specified replication units] while satisfying the locked in-support equivalence margin.”
- **Tail-only:** “The mean changed little; the supported benefit was confined to the predeclared high-error tail.”
- **Equivalent but no tail benefit:** “The guard preserved the backbone within the locked margin but did not provide evidence of useful tail control.”
- **Seed-specific:** “The apparent gain depended on particular training draws and does not support a general safeguard claim.”
- **Detector not superior:** “Spread is retained as a transparent internal signal, not as the best error detector.”
- **Untouched test neutral:** “The one-shot test did not confirm a benefit; it bounds the method's transportability.”
- **Untouched test adverse:** “The locked safeguard failed the preservation criterion; positive deployment claims are withdrawn.”
- **Planner neutral/adverse:** “No operational safety-efficiency benefit was demonstrated; the intervention output remains a design proposal.”

## Integration order

1. Freeze and archive configuration/code before any untouched labels are inspected.
2. Complete A1-A3 audits and all development/exploratory runs.
3. Finalise figures/tables for RQ1-RQ3 without changing the untouched protocol.
4. Execute the untouched test once and preserve its raw log/checksums.
5. Complete the planner/latency study if feasible under the frozen policy.
6. Integrate prose into `main.tex`, using the conditional language above.
7. Compile, inspect every page, verify hyperlinks/appendix prefixes, and cross-check every number against a released artifact.
8. Fill the referee response with final page/line references only after the PDF is stable.
