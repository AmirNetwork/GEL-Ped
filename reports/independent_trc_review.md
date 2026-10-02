# Independent TR-C-style scientific review of GEL-Ped

**Review namespace:** referee3  
**Review date:** 2 October 2026 (Australia/Sydney)  
**Frozen evidence:** commit `73b945b`, configuration `configs/referee2_lock.json`, and all `referee2_*` outputs  
**Current recommendation:** **Not ready for submission; major scientific revision required**

## 1. Review scope and evidence boundary

This review treats commit `73b945b` as an immutable baseline. It evaluates the
manuscript as a new submission rather than defending earlier drafting choices.
No numerical result in the frozen `referee2_*` namespace may be overwritten or
silently reinterpreted. New decisions, configurations, scripts, predictions,
tables, and figures must use a `referee3_*` namespace and must record whether a
calibration, development, or test result was visible when the decision was
made.

The latest external report recommends major revision. Its central criticism is
scientific rather than editorial: GEL-Ped is described as a safeguard against
rare neural instability, but the paper is organised around mean RMSE; the
crossing result is development-informed and partly driven by one divergent
member; and the clean ETH/UCY comparison is effectively neutral. The report's
suggested wording, titles, datasets, and references are advice, not evidence.
They should be adopted only where they improve the scientific argument.

## 2. What the frozen evidence currently supports

The baseline evidence supports a narrow claim. A five-member graph ensemble is
left unchanged whenever its smoothed local disagreement remains inside a
complete-run calibration envelope. Outside that envelope, a one-sided
exponential rule shrinks its prediction towards a bounded feature-based HGB
fallback. In the development-informed crossing stress test, the guarded
predictor reduces run-macro RMSE from 0.35427 to 0.34627 m/s and improves all 13
runs. In the highest reported instability stratum, RMSE falls from 0.56536 to
0.46653 m/s. In familiar, altered, bottleneck, and unidirectional regimes, the
guarded result remains close to the graph ensemble.

That evidence does **not** establish a general improvement under flow shift.
The crossing mean is influenced by run `d_1`, where one member reaches 0.8396
m/s. Without `d_1`, the mean reduction is about 1.3%. The feature-based HGB is
slightly better than GEL-Ped in crossing, and the clean five-scene ETH/UCY
comparison is approximately neutral. At 3.2 s, constant velocity and HGB beat
recursive GEL-Ped. Nominal 90% prediction regions cover only about 36--52% in
crossing. These are scope boundaries, not details to hide.

The manuscript's potentially publishable contribution is therefore a
transparent runtime-assurance mechanism and an evaluation of when ensemble
disagreement can and cannot protect a pedestrian predictor. The most useful
scientific finding may be the boundary itself: disagreement can expose
variance-type failures across trained members, but it cannot expose shared
bias that moves the whole ensemble in the same wrong direction.

## 3. Strengths that should be preserved

1. **Complete-run separation.** Physical runs, not frames, are the unit of
   splitting and run-level RMSE is computed before aggregation.
2. **Declared chronology.** Crossing is explicitly identified as
   development-informed rather than presented as an untouched confirmation.
3. **Auditable guard.** The one-sided exponential has two calibrated constants
   and gives exactly (g=1) whenever (E\leq\tau).
4. **Distribution-aware reporting.** Per-run results, ranges, medians,
   leave-one-run sensitivity, and the `d_1` diagnosis are unusually candid.
5. **Negative evidence.** The ETH/UCY null result, HGB crossing result,
   long-rollout disadvantage, and failed coverage audit are reported.
6. **Reproducible assets.** Code, partitions, processed tables, figures, and a
   frozen snapshot exist and the baseline test suite previously passed 38
   tests.

These strengths are valuable only if the referee-3 work preserves their
discipline. In particular, a new held-out dataset cannot become another tuning
set after its first inspection.

## 4. Blocker A: internal consistency and fairness

These issues arise from the handoff audit and must be resolved before new
performance claims are considered.

### A1. Interaction-decay mismatch

The manuscript and controlled Juelich pipeline use an interaction decay length
of 0.8 m, whereas `src/pedgeom/external.py` passes 0.35 for ETH/UCY. The claim
that the same representation is used across datasets is therefore not
literally true. Choosing whichever value improves ETH/UCY after seeing the
test results would be test-set tuning.

**Required resolution:** establish the provenance of 0.35, run a frozen
consistency audit under calibration/validation rules, and either standardise
the representation before test evaluation or disclose a pre-existing,
scientifically justified dataset-specific scaling. Record both variants in
`referee3_*` outputs and do not overwrite the neutral frozen comparison.

### A2. Guard memory is indexed by retained rows, not necessarily time

The paper describes five forecast origins as a maximum two-second memory.
However, controlled evaluations retain at most 6,000 sampled forecasts per run.
If smoothing occurs after subsampling, five retained rows may span irregular
or substantially longer intervals.

**Required resolution:** audit frame gaps and realised temporal spans for every
regime. Prefer smoothing on all causal origins before evaluation subsampling or
implement an explicitly time-based window. Add irregular-sampling and
scene-reset tests. Until this is resolved, the two-second claim is unsupported.

### A3. Gate controls are not matched to the same expert pair

The primary guard combines graph and coarse HGB, while the Mahalanobis and
learned-gate controls appear to combine graph and prospective-geometry HGB.
Those rows change both the gate and the fallback, so they do not isolate gate
quality.

**Required resolution:** compare instability, Mahalanobis, supervised gate,
and graph-fallback disagreement using the identical graph/coarse-HGB expert
pair, data, seeds, and tuning budget. Tune gates on calibration data only.
Retain geometry-fallback variants only as separately labelled ablations.

## 5. Blocker B: the tail-control premise is not yet tested directly

Mean RMSE must remain visible because it was the earlier primary endpoint, but
it is not an adequate primary lens for a method intended to suppress rare
failures. The referee-3 analysis plan must be frozen before any new held-out
shift is inspected and should include:

- per-run and per-regime 90th, 95th, and 99th error quantiles;
- CVaR at 95% and 99%;
- the rate of velocity error above 0.5 m/s, corresponding to 0.2 m over the
  0.4 s target horizon;
- AUROC and AUPRC for detecting the top 5% of graph errors with local spread,
  Mahalanobis distance, and graph-HGB disagreement;
- risk-coverage curves and area under the risk-coverage curve; and
- HGB and an oracle min(graph,HGB) in every instability-stratum plot.

These analyses must use forecast-level outputs but respect run dependence in
uncertainty estimates. Results derived from already inspected crossing data
are exploratory; the endpoint definitions, directions, and multiplicity plan
must be fixed before the new held-out test.

## 6. Blocker C: training randomness is an unmeasured replication axis

All 13 crossing runs were scored with one trained five-member ensemble. Runs
are not independent replicates of the training event that produced the
divergent member. The current sign-flip result therefore conditions on one
particular ensemble draw.

At minimum, train a pool of 25--50 independent members, form at least 20
five-member ensembles, and recalibrate the envelope for each draw. Report the
distribution across draws of mean and median crossing gain, tail metrics,
ETH/UCY macro change, guard activation, and the frequency and definition of a
divergent member. Sweep (M\in\{3,5,10,20\}). Inference should reflect both
training-draw and run variation, using crossed random effects where stable or
transparent draw-wise distributions and cluster/bootstrap intervals where it
is not.

A result that disappears under ordinary retraining cannot support a general
runtime-safeguard claim. A result that persists primarily in high-spread tails
across draws would materially strengthen the paper.

## 7. Blocker D: targeting must be separated from blending

Because HGB is slightly better than the graph ensemble in crossing, any amount
of graph-to-HGB blending may improve the crossing mean even if the gate does
not identify risky forecasts. The current evidence does not fully separate
these mechanisms.

Required controls are:

1. a fixed global graph/HGB blend with its weight selected on calibration
   runs;
2. an oracle descriptive fixed blend whose weight equals each regime's
   realised mean (g);
3. within-run permutation of (g), preserving its distribution while breaking
   forecast-level targeting;
4. a causal speed gate and partial association of (E) with error after
   conditioning on speed; and
5. per-stratum graph, GEL-Ped, HGB, fixed blend, permuted-(g), and oracle
   results.

The manuscript may call the mechanism targeted only if it beats the matched
fixed and permuted controls in the high-instability region. Otherwise it should
be described as adaptive blending.

## 8. Blocker E: variance failures and shared-bias failures must be separated

Moving from familiar corridor flow to crossing raises graph RMSE by roughly
0.19 m/s, while GEL-Ped recovers only about 0.008 m/s. Most of the degradation
is therefore not repaired by disagreement-based guarding. A useful revision
should decompose squared error into error of the ensemble mean and dispersion
across members, by regime and instability stratum, and quantify the fraction
of shift-induced error that the monitor can observe.

This decomposition should become the conceptual spine of the paper. GEL-Ped
addresses detectable variance-type failures. Shared bias requires different
tools, such as broader training flows, causal or physics constraints, online
adaptation, or model-class diversity. This distinction is more transferable
than a small average gain and prevents the title and abstract from implying a
general shift solution.

## 9. Blocker F: there is no positive, untouched shift confirmation

Crossing informed development. The Tordeux data informed the final comparison.
ETH/UCY is cleaner but shows no meaningful advantage and contains too few
scenes for a powerful exact test. A new test must therefore be selected using
scientific suitability and data availability before its outcomes are viewed.
A T-junction, corner, genuine naturalistic cross-flow, or pedestrian-vehicle
intersection set is appropriate if its coordinate, timing, and identity data
support the frozen causal protocol.

The full pipeline, endpoints, equivalence margin, and multiplicity plan must be
tagged before the first test result is computed. The new set may be evaluated
once. “No harm” must be tested with two one-sided equivalence tests at a
pre-stated practical margin, not inferred from a non-significant difference.
Pedestrian-level models should use run/scene and pedestrian grouping so that
frame dependence is not treated as replication.

If the new held-out result is neutral, the paper may still be publishable as a
carefully bounded negative/scope study, but not as evidence of broad accuracy
improvement. If it is adverse beyond the equivalence margin, submission should
stop until the mechanism or claim is redesigned on development data and a new
untouched confirmation is available.

## 10. Important evidence beyond the six essential blockers

### 10.1 Causal lagged-error competitor

At a 0.4 s horizon, realised expert errors become available quickly. A causal
exponential-weighting or fixed-share baseline should update graph/HGB weights
using only outcomes available at time (t), at pedestrian, neighbourhood, or
site level. GEL-Ped's distinct potential value is immediate response for a new
pedestrian before enough labels accumulate. Their combination should also be
tested under a calibration-locked rule.

### 10.2 Architecture and training-recipe robustness

Wrap at least one established compact predictor under the same 0.8/0.4 s
protocol and apply the same guard. Test input standardisation, held-run early
stopping, weight averaging, and a larger training pool. If ordinary training
hygiene eliminates divergent members, the paper must report that finding. MC
dropout, Gaussian-NLL ensembles, and snapshot ensembles are useful secondary
signals, but they should not displace the essential replication analysis.

### 10.3 Downstream transport value

The cleanest path is a reactive-layer claim rather than a long-horizon one.
Use a replayed-crowd local planner with approximately 0.4--1 s look-ahead and
compare a fixed safety margin against a pre-specified margin controlled by
(E) or (g). Report personal-space intrusions, minimum separation, travel
time, stop frequency, and failure rate. Also report CPU latency for the five
members, HGB, and guard at a stated pedestrian count. Do not call the existing
94%-prevalence collision screen a safety demonstration.

### 10.4 Calibration audit

State which models are cross-fitted and which are deployed. Report the realised
familiar fraction above (	au), run-bootstrap uncertainty for (	au,S), origin
spacing, whether smoothing precedes forecast subsampling, sensitivity to
windows 1/3/5/10, and sensitivity to exponential, linear, and logistic gates.
Report held-scene activation rates. Treat the prior-velocity axis as
co-primary for deployment claims. Remove zeroed wall variables or evaluate a
real wall-aware variant.

### 10.5 Predictive regions

The current interval experiment is a negative result. The minimum defensible
revision is to remove it from the contribution set and retain one concise
limitation. A stronger optional extension is spread-normalised plus adaptive
conformal inference using delayed coverage errors, reporting both coverage and
region size by instability stratum. It must not be presented as confirmatory
unless its rule is fixed before the untouched test.

## 11. Literature, writing, and presentation

The related-work section should position GEL-Ped as predictor-level runtime
assurance across three strands: pedestrian/hybrid prediction, uncertainty
under distribution shift, and safety assurance with delayed feedback. A compact
positioning table should compare fixed hybrids, adaptation, test-time training,
OOD detection, learned gates, selective prediction, online expert weighting,
adaptive conformal methods, and GEL-Ped by label requirement, identity inside
the envelope, output type, and trigger signal. Suggested references from the
report must be bibliographically verified, and three to five recent relevant
TR-C papers should anchor the transport context.

The revision should be organised around four questions: detection, action,
scope, and use. Use “calibration envelope” consistently; call the fallback
“feature-based HGB”; remove revision-history language; resolve symbol clashes;
define all terms before equations; and keep caveats in the introduction and
conclusion rather than repeating them. Figures should separate detection,
tail, and rollout evidence. A data-role timeline and the cross-fitting loop
should be visually explicit.

## 12. Leakage-controlled execution order

1. Reproduce the frozen tests and record environment details without changing
   `referee2_*` evidence.
2. Resolve A1--A3 and add tests for time indexing, feature consistency, and
   matched expert pairs.
3. Implement all metrics and baselines on calibration/development data.
4. Freeze `configs/referee3_lock.json`, endpoint definitions, equivalence
   margins, multiplicity, ensemble draws, and held-out dataset inclusion rules.
5. Create an immutable tag and checksum manifest.
6. Run the untouched test once and preserve forecast-level outputs.
7. Draft the manuscript from the locked evidence, not from the desired story.
8. Run the complete test suite, regenerate all `referee3_*` tables/figures,
   compile LaTeX, and inspect every PDF page.

## 13. Readiness gates

### Current state: not ready

The frozen paper would likely receive another major-revision or rejection
decision because the main gain is development-informed, training-draw
sensitivity is unknown, targeting is not isolated from blending, and transport
use is asserted rather than demonstrated.

### Minimum state for a defensible resubmission

- A1--A3 resolved without test-informed parameter selection.
- Tail, detection, and risk-coverage endpoints complete.
- At least 20 independent five-member draws and the ensemble-size sweep.
- Fixed-blend, permuted-(g), speed-gate, and lagged-error controls.
- Bias/variance scope decomposition.
- Formal in-support equivalence.
- One frozen, untouched shift evaluation.
- A credible reactive-planner demonstration and latency audit, or a
  correspondingly narrower paper that removes operational claims.
- A final manuscript whose claims match all positive and negative evidence.

### Stop conditions

Do not recommend submission as a positive method paper if any of the following
holds:

- the tail advantage is absent across independent ensemble draws;
- fixed or permuted blending explains the high-instability improvement;
- the untouched test is worse than the pre-stated equivalence bound;
- the guard score does not detect high-error forecasts better than simple
  speed or disagreement controls; or
- an operational claim remains unsupported by a downstream experiment.

In those cases, the scientifically sound alternatives are to recast the paper
as an empirical study of disagreement-monitor limits or to perform new method
development on explicitly designated development data before acquiring a new
confirmation set.
