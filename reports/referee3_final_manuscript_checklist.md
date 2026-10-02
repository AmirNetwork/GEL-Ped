# Referee-3 final manuscript integration checklist

**Scope.** This checklist maps the 2 October 2026 referee report to the current
`manuscript/overleaf_trc/main.tex`, the round-3 response skeleton, the bibliography,
and `configs/referee3_lock.json`. It is intentionally result-blind: bracketed values
must come only from the final frozen artifacts. Line numbers refer to the current
`main.tex` snapshot and will move during integration.

## Executive decision

The manuscript now has a credible, accessible front half. Preserve the story and
most of the prose in the title/abstract (lines 28--34), Introduction (44--54),
Related work and gap (63--101), Method (103--200), and Experimental design
(203--226), subject to the targeted edits below.

Do **not** incrementally edit the present Results. Replace the complete old evidence
block--Results, Discussion, and Conclusion (lines 228--334)--after the referee-3
artifacts are signed. Most of the appendices (351--520) also encode the superseded
pipeline and should be rebuilt rather than retained. The final paper should be a
runtime-safeguard paper whose central claim is conditional:

> Ensemble disagreement can reveal and temper variance-type neural failures; it
> cannot diagnose error shared by the model family.

Mean accuracy remains visible for continuity, but tail detection, targeting versus
mere blending, retraining stability, untouched-shift performance, and reactive
planner consequences determine whether that claim is supported.

## Recommended final narrative and document structure

1. **Introduction** (`sec:introduction`): a short-horizon prediction is useful
   because it can change the next robot/vehicle action; crowd interaction makes
   rare corrections consequential; the operational gap is between warning and
   usable forecast; disagreement has a clear shared-error blind spot.
2. **Related work and research gap** (`sec:related`): pedestrian/hybrid prediction;
   uncertainty and robustness under shift; runtime assurance and delayed online
   feedback; then the positioning table.
3. **GEL-Ped method** (`sec:method`): task and anchor; causal interaction cues;
   graph ensemble and feature-based HGB fallback; time-indexed guard; cross-fitting
   and deployment. Keep the explanation in physical language before equations.
4. **Experimental design** (`sec:experiments`): data roles and test firewall;
   matched comparators; ensemble/backbone replication; endpoints and inference;
   reactive planner protocol.
5. **Results**, ordered by the four questions already posed in lines 48--54:
   - **5.1 Detection:** AUROC/AUPRC, risk--retention, and score confounding.
   - **5.2 Action:** tail errors and targeting versus fixed/permuted blending.
   - **5.3 Scope:** shared ensemble error versus member variance; retraining,
     ensemble-size, backbone, untouched-shift, and equivalence evidence.
   - **5.4 Use:** lagged expert weighting, cold start, reactive planner, and latency.
6. **Discussion**: operational interpretation; when to trust/intervene/recalibrate;
   the shared-error blind spot; limits of replay and the 0.4-s task.
7. **Conclusion**: one qualitative paragraph stating what was learned and one
   sentence delimiting scope. Do not repeat a catalogue of experiments.

This ordering lets a non-specialist follow **problem -> warning -> action -> limits
-> transport consequence**. It also prevents the old 2.3% crossing mean from
becoming the story.

## Claim gate before prose is written

| Proposed claim | Minimum evidence required | If the gate fails |
|---|---|---|
| Spread *detects* difficult forecasts | AUROC/AUPRC and risk--retention better than speed, Mahalanobis, and graph--HGB disagreement on the relevant regimes | Say spread is an intervention heuristic; do not call it an error detector |
| GEL-Ped *targets* failures | Tail benefit over both calibration-fixed and within-run permuted weights, using the same graph/HGB predictions | Use “adaptive blending,” not “targeted safeguard” |
| Benefit survives retraining | Effect distribution over 30 draws, crossed run/draw uncertainty, and ensemble-size sensitivity | Scope all claims to the observed training draws |
| Neural accuracy is preserved in the calibration envelope | TOST confidence interval lies within the pre-stated 0.002 m/s and 1% margins | Report the difference descriptively; never infer equivalence from non-significance |
| Guard is backbone-general | The locked transformer-style backbone shows the same qualitative detection/action mechanism | Say the mechanism was tested on one primary architecture and one inconclusive sensitivity |
| Untouched shift evidence supports transfer | One-shot T-junction family after Holm correction, reported exactly as frozen | State a null/adverse held-out result and narrow the operating scope; do not tune |
| The intervention has operational value | Paired fixed/adaptive replay improves a safety metric with an explicit efficiency cost and stated event unit | Present `g` only as a diagnostic signal, not as demonstrated planner value |
| GEL-Ped supplies calibrated uncertainty | Not claimed under the selected “cut” decision | Keep only a one-sentence limitation; no coverage table or safety-region language |

## Referee-point integration map

### M1. Tail metrics must be primary evidence for the safeguard

- **Already in place:** endpoint definitions at lines 219--224 (`sec:experiments`).
- **Final main-text destination:** Results 5.1 and 5.2; a compact table
  `tab:tail-targeting` with q90/q95/q99, CVaR95/CVaR99, and error >0.5 m/s for graph,
  GEL-Ped, HGB, fixed blend, and oracle; `fig:detection` for risk--retention plus
  AUROC/AUPRC.
- **Required artifacts:** `referee3_tail_metrics.csv`,
  `referee3_detection_metrics.csv`, and `referee3_risk_retention_curves.csv` as
  listed in the response skeleton.
- **Writing constraint:** call crossing/ETH/UCY/bottleneck/unidirectional tail
  results exploratory; identify the frozen T-junction family as confirmatory.
- **Confounding check:** report speed as a competing detector and the association
  between spread and error conditional on observed speed. Define any forecast-level
  crowding proxy explicitly; do not relabel facility-average density as local
  exposure.
- **Remove/replace:** old mean-led opening and Table `tab:controlled` (231--251),
  old Figure `fig:evidence` (280--285), and the unsupported “small neural tail”
  conclusion at 276.

### M2. Training draw is a replication axis

- **Already in place:** 25-member pool, 30 draws, and M={3,5,10,20} in lines
  214--217 and the lock file.
- **Final main-text destination:** Results 5.3; `fig:replication` showing distributions
  across draws, not a single seed set; one row block in `tab:robustness` for ensemble
  size, divergent-member frequency, and crossed run/draw intervals.
- **Required artifacts:** training manifest, draw calibration, draw-run metrics, and
  draw-run inference. Report both mean and median effects across draws.
- **Remove/replace:** the single-ensemble exact-test interpretation at 273--276,
  the d_1/member-3 anecdote as central evidence, and every single-draw number in
  `tab:crossing-runs` (390--417). If d_1 is retained, use it only as an illustrative
  case after the replicated result.

### M3. Targeting must be separated from generic blending

- **Already in place:** matched controls described at lines 214--215 and frozen
  permutation settings in the lock.
- **Final main-text destination:** Results 5.2; `tab:tail-targeting` and a simple
  strata panel in `fig:targeting` containing graph, HGB, GEL-Ped, calibration-fixed
  blend, realised-weight descriptive blend, within-run permuted gate, and oracle.
- **Decision rule:** use “targeting” only if GEL-Ped beats the fixed and permuted
  controls in the high-instability tail. Otherwise revise the paper around adaptive
  expert blending.
- **Remove/replace:** old control table `tab:controls` (451--470), whose rows change
  both gate and fallback, and the claim at 472 that intervention is directly
  attributable to instability.

### M4. Separate observable member variance from shared ensemble error

- **Already in place:** the conceptual distinction at lines 52 and 100--101 and the
  planned decomposition at line 224.
- **Final main-text destination:** Results 5.3, preferably a two-panel decomposition
  within `fig:replication` or a small `tab:decomposition`; Discussion first
  subsection.
- **Terminology:** report the exact squared-error identity as **shared ensemble
  error** and **between-member variance**. Do not call the empirical shared term
  population “bias squared” unless the estimand is formally defined.
- **Required interpretation:** quantify what fraction of the *change* from familiar
  to each shifted regime is visible in between-member variance, with uncertainty.
  Explain that adaptation, broader context, or a more diverse family is needed for
  the remainder.
- **Remove/replace:** any inherited “96% shared bias” number unless reproduced by
  the corrected pipeline.

### M5. Untouched shift and formal no-harm assessment

- **Already in place:** untouched T-junction role at lines 205--210; equivalence
  margins and run-level inference at 219--222; frozen four-endpoint Holm family in
  the lock.
- **Final main-text destination:** Figure 1 (`fig:experiments`) must become a true
  data-role timeline; Results 5.3 should include `tab:heldout-equivalence` with the
  one-shot T-junction endpoints, adjusted p values, confidence intervals, and the
  TOST result for in-envelope/familiar cases. A pedestrian-level mixed-effects
  sensitivity, with run/scene and pedestrian grouping stated, belongs in the lean
  statistical appendix; it complements rather than replaces complete-run inference.
- **Writing constraint:** report null or adverse held-out evidence without tuning or
  euphemism. “No significant difference” is never evidence of preservation.
- **Remove/replace:** 231 (“effectively unchanged”), 288 (“does not support” may stay
  only as an external null), 321 (“within 0.0003”), 328, and 334. Replace with the
  frozen equivalence/held-out outputs.

### M6. Causal delayed-error expert weighting

- **Already in place:** comparator definition at 214--215; delay, learning-rate,
  fixed-share, and cold-start rules in the lock.
- **Final main-text destination:** Results 5.4, a concise `tab:online-expert` with
  overall and first-1.6-s performance for graph, HGB, GEL-Ped, delayed weighting,
  and any frozen combination.
- **Interpretive question:** does spread help before a new pedestrian has enough
  labelled history, and does delayed feedback later catch shared failures?
- **Remove/replace:** the future-label oracle paragraph at 312 as the main online
  comparator. The oracle may appear only as a clearly unattainable descriptive
  ceiling in an appendix note.

### M7. Architecture/training robustness and fallback terminology

- **Already in place:** the second-backbone protocol at line 217 and the complete
  primary graph/HGB specification at 149--172. The current text correctly uses
  “feature-based HGB fallback.”
- **Final main-text destination:** Results 5.3, `tab:robustness`, with graph versus
  transformer-style backbone, training recipe, divergence frequency, detection,
  and action results. State standardisation, held-run early stopping, and checkpoint
  averaging once in Experimental design.
- **Scope rule:** do not claim architecture independence unless the completed second
  backbone supports the mechanism. MC dropout, NLL heads, snapshots, CV fallback,
  and median fallback should be reported only if frozen artifacts exist; otherwise
  acknowledge them as untested variants in one sentence, not as “addressed.”
- **Bibliographic anchor:** cite the actual backbone precedent beside its
  description. `mohamed2020`, `salzmann2020`, and recent transformer entries are
  available, but a generic “compact transformer” is not itself an established
  published benchmark.
- **Remove/replace:** every “coarse HGB” occurrence from 231 onward; old capacity
  table `tab:capacity` has the obsolete 13,395 parameter count and should be removed
  or regenerated at 13,203.

### M8. Demonstrate the reactive-layer transport use

- **Already in place:** the manuscript motivates immediate robot/vehicle action at
  44--48; the lock freezes an open-loop paired replay at 0.4-s replanning and 0.8-s
  look-ahead.
- **Final main-text destination:** add the planner protocol to Experimental design;
  Results 5.4 should contain `tab:planner-latency` (intrusions, minimum separation,
  travel time, stops, failures, and post-prediction CPU latency) and, only if useful,
  a small safety--efficiency plot `fig:planner`.
- **Mandatory caveat:** recorded pedestrians do not react to the robot; this is a
  paired replay, not a reciprocal simulation or safety proof.
- **Remove/replace:** the old rollout-centred Figure 3D and 3.2-s claims at 316. A
  one-line diagnostic may remain in the appendix, but the paper must not imply that
  recursive one-step prediction is a competitive long-horizon forecaster.

### M9. Retire the high-prevalence collision screen

- **Final main-text destination:** fold the replacement into M8. Define the event
  unit, physical intrusion threshold, closest-approach calculation, and base rate;
  report continuous minimum separation/TTC where available.
- **Remove, do not archive in the manuscript:** the complete paragraph at 310,
  including 73,051/4,949 counts, confusion matrices, miss/false-alarm rates, and the
  sentence about coincidental rounding. Keeping it in a “historical sensitivity”
  appendix would add clutter and revive a retired claim.

### M10. Calibration and design clarifications

- **Deployed vs cross-fitted:** add a four-step training/calibration/deployment box
  after line 183 or to Figure 2 (`fig:architecture`). Explicitly distinguish six-run
  OOF models used to estimate the envelope from all-calibration-run deployed fits;
  report realised `S>tau` rate for each familiar/held scene.
- **Inside-envelope wording:** lines 34, 106, and 194 are correct. Preserve “weight
  equals one whenever spread is inside the calibration envelope,” not “supported
  conditions imply g=1.”
- **Threshold uncertainty:** add run-bootstrap intervals for tau and IQR scale in
  Results 5.3 or the concise robustness table.
- **Window:** line 181 now states current plus up to four prior origins over a trailing
  1.6 s. Add activation by track age/cold start and 1/3/5/10-origin sensitivity; do
  not call them “observations.”
- **Functional form:** line 194 properly freezes exponential and identifies linear/
  logistic sensitivities. Put the weight-map/calibration-distribution plot in the
  appendix only if it remains legible and informative.
- **ETH/UCY:** line 210 states the two-interval anchor. Add validation-provenance and
  scene firing-rate results; include smoothed CV only if its locked artifact exists.
- **Axis:** line 115 correctly makes prior velocity primary and entrance direction a
  sensitivity. Delete the old appendix table in which entry axis is primary.
- **Walls:** lines 115 and 158 correctly exclude wall channels. If a with-walls result
  exists, label it facility-specific and keep it in the compact robustness appendix.
- **Sampling:** line 212 is clear; add the fixed seed/manifest reference and avoid a
  paper table whose “Forecasts” column conflates the 6,000 training cap with all
  chronological evaluation origins.

### M11. Cut the failed interval component

- **Already in place:** no positive interval claim appears in the new abstract or
  contribution paragraph; adaptive conformal work is correctly positioned as a
  different delayed-feedback tool at lines 73--76.
- **Final destination:** one Discussion sentence: the intervention score is not a
  calibrated error probability or coverage guarantee, and adaptive regions are
  future work.
- **Remove:** the detailed coverage block at 328 and the full uncertainty paragraph
  at 493. Do not publish the 35.9--52.2% table/radii as though it were part of the
  revised contribution.

## Literature, gap, and contribution checks

### What is already strong

- Lines 44--52 give a readable operational story without assuming knowledge of
  physics or neural networks.
- Lines 65--76 engage pedestrian dynamics, uncertainty under shift, runtime
  assurance, selective prediction, and delayed feedback.
- `tab:positioning` separates detection from action and avoids claiming that hybrid
  weighting itself is novel.
- Lines 100--101 connect the literature directly to the monitor--action--blind-spot
  gap.
- Haghani is cited for empirical methods/validity (`haghani2018`, `haghani2023`),
  which is relevant and not promotional.

### Final improvements

1. Compress line 54 from four inventory-style contributions to three learned
   contributions: (i) identity-preserving calibrated action, (ii) leakage-resistant
   safeguard evaluation, and (iii) evidence delimiting detectable variance from
   shared error. Planner output is an implication, not a fourth novelty claim.
2. Expand `tab:positioning` just enough to keep **fixed physics/hybrid**, **learned
   gates**, **runtime monitoring**, and **delayed online adaptation** distinct. The
   current four-row version merges too much. Use available verified references such
   as `bahari2021`, `yue2022`, `korbmacher2024hybrid`, and
   `lakshminarayanan2017`; avoid a sprawling survey table.
3. Cite one relevant recent journal paper per application thread, not the journal's
   name. The current `liu2024`, `li2023cyclist`, `mai2025`, and `munir2025` entries
   already do this subtly. `haghani2024trc` can support crowd movement only if its
   statement is directly relevant.
4. The bibliography currently resolves every citation in `main.tex`. Before final
   export, verify publisher metadata/DOIs for the entries flagged by the referee
   (`kim2024`, `lakshminarayanan2017`, `messaoud2025`, `xu2022tgnn`) rather than
   inventing identifiers.
5. Continue to omit the unpublished `ghorbani2022` from the argument except, if
   essential, one neutral sentence explaining conceptual lineage. Do not discuss
   its limitations or present it as prior validation.

## Accessible prose and terminology

- Use one term for each concept: **calibration envelope**, **instantaneous spread
  s**, **smoothed disagreement S**, **graph ensemble**, **feature-based HGB
  fallback**, **graph weight g**, **shared ensemble error**, and **between-member
  variance**.
- Reserve “bounded” for the explicit 3 m/s output cap and the finite HGB leaf-value
  range. It is not a behavioural or safety guarantee.
- Prefer “short-horizon pedestrian velocity prediction” to “crowd prediction.”
  “Crowd flow” remains appropriate for the operating context.
- Introduce each equation with its physical meaning. Keep implementation epsilons
  (currently line 145) in a short reproducibility note rather than the conceptual
  method.
- Avoid repeated caveats. State the shared-error scope once in the Introduction,
  interpret it in Discussion, and close with it once in the Conclusion.
- Use “initialisation” or “initialization” consistently. Use “Juelich” in prose if
  ASCII portability is preferred, while retaining official DOI titles verbatim.
- The abstract should remain high-level as requested by the author. Add at most one
  compact, verified findings sentence; do not turn it into a list of q-values,
  p-values, and dataset acronyms.

## Minor/editorial closure grid

| Referee item | Current hook | Final action |
|---|---|---|
| Task name/title | 28, 34 | Keep “pedestrian prediction”; avoid “crowd prediction” as the task name |
| `g=1` wording | 34, 106, 194 | Keep the conditional “whenever spread is inside the calibration envelope” |
| Five past “observations” | 181 | Keep “current and up to four preceding forecast origins in the trailing 1.6 s” |
| Graph detail | 149--163 | Widths, attention pooling, head, loss, cap, and 13,203 parameters are now sufficient |
| ETH two-frame anchor | 210 | State annotation-noise limitation; include smoothed CV only if frozen |
| Symbol clashes | 115--135, 174--194 | Current `r_b`, residual deltas, `e_i`, `s_i`, and `S_i` resolve the clashes; check figures too |
| Numerical epsilon | 145 | Move the `10^-8` detail to a short implementation/reproducibility note |
| Wall inputs | 115, 158 | Primary model correctly removes them; do not list zero-valued channels |
| Equation rendering | 156 | `\quad M=5` is now correct; verify rendered PDF |
| Run-balanced loss | 158--162 | Definition is now explicit; retain it |
| Capacity sweep | old 474--491 | Remove; if retained, regenerate seed uncertainty and parameter counts |
| Meaning of bounded fallback | 172 | State cap/leaf-range meaning; never imply behavioural or safety boundedness |
| Prospective closest-approach encoder | absent from new Method | Keep absent unless a clearly motivated, frozen ablation exists |
| Cold-start activation | 181 and Results 5.4 | Add activation/firing rate by time since first usable origin |
| Quantile convention | 183--187 | Correctly explicit; add run-bootstrap uncertainty |
| Neutral chronology | 205--212 | Keep role-based language and add frozen tag/time to Figure 1, not review history |
| 16-Hz comparability | 208 | Current causal 10-Hz projection supersedes 0.8125/0.375 text; report validation of the resampling |
| 6,000 cap | 212 | Add exact seed/manifest; distinguish fitting cap from all-origin evaluation |
| “Predeclared” | 222 and old 421 | Use “frozen before one-shot evaluation” with tag/registry; remove the old 15-test phrase |
| High-instability stratum | 224 | Definition is correct; add the share of forecasts in every reported regime |
| ETH low weight | Results 5.1/5.3 | Report scene firing rates and speed/crowding/timing context, not an unexplained mean `g` |
| Future-error oracle | 215 and Appendix C | Define as target-using and unattainable; never call it an online method |
| Recursive propagation | 226 | If retained, state that each expert/spread/weight is recomputed; keep it diagnostic and secondary |
| Publication/review language | old 310, 334, filenames | Delete completely from manuscript |
| Appendix prefixes | macros 21--24, 351 onward | Keep hyperlink macros; check rendered headings for one “Appendix” prefix only |
| Local density | old 356 | Keep facility density clearly global; use a defined forecast-level measure for stratification |
| Repository statement | 339--340 | Replace promises with neutral release tag, licence, archive DOI, and entry point once real |
| DOI metadata | bibliography | Verify flagged records against primary publisher/proceedings pages before export |

## Figures and tables: exact final roles

1. **Figure 1, `fig:experiments`:** regenerate and rename the source file (the current
   `experimental_context_referee2.pdf` exposes review history). Keep the two real
   experiment images, but make the chronology/data-role timeline dominant and mark
   calibration, development-informed exploration, external transfer, and untouched
   T-junction confirmation.
2. **Figure 2, `fig:architecture`:** preserve the clean method diagram and add the
   OOF cross-fitting loop that produces tau/IQR, followed by the distinct deployed
   refit. Match manuscript fonts and use clickable text reference via `\figref`.
3. **Figure 3, `fig:detection`:** risk--retention plus detector comparison. Do not
   mix it with rollout curves.
4. **Figure 4, `fig:targeting`:** error by instability stratum with graph, GEL-Ped,
   HGB, fixed blend, permuted gate, and oracle; show regime shares.
5. **Figure 5, `fig:replication` or `fig:planner`:** use only if it makes draw
   robustness or safety--efficiency trade-off clearer than a table. Avoid a six-
   figure paper if the same evidence fits cleanly in a table.

Main tables should be few and argumentative: positioning; notation; data roles;
tail/targeting; held-out/equivalence/robustness; planner/latency. Use `\tabref`,
`\figref`, `\secref`, and `\appref` for every first and interpretive reference.
Report differences from the graph and draw uncertainty directly instead of forcing
the reader to subtract cells.

## Mandatory old-result removal manifest

The following content cannot survive the corrected target-blind, prior-velocity,
wall-free, resampled, multi-draw pipeline unless independently regenerated:

- **Lines 231--334 in full:** all controlled/additional/ETH values, p values, g
  values, d_1 narrative, collision screen, future-label oracle, rollout values,
  discussion percentages, and conclusion claims.
- **Source filenames at lines 58 and 282:** anything containing `referee2`.
- **`tab:controlled`, `tab:additional`, `tab:ethucy`:** all numerical cells and old
  captions; the 16-Hz 0.8125/0.375 caption is now false because the locked pipeline
  performs causal 16-to-10-Hz projection.
- **`tab:crossing-runs`:** conditioned on one five-member draw; replace with draw
  distributions and release per-run values externally.
- **`tab:holm`:** the old 15-test family. The new confirmatory family is the four
  held-out endpoints in the lock; other regimes are exploratory.
- **`tab:controls`:** expert pairs are not matched and the interpretation is
  confounded.
- **`tab:capacity`:** 13,395 is obsolete after wall removal (current method: 13,203)
  and a capacity grid adds little to the final story.
- **Uncertainty paragraph at 493:** removed under the M11 cut decision.
- **`tab:encoding`:** contradicts the locked prior-velocity primary axis and refers
  to a future-label filter that no longer exists.
- **Appendix run counts/“Forecasts” at 369--385:** recompute membership after the
  corrected causal history join and distinguish the training cap from the complete
  evaluation set.
- **Phrases to eliminate globally:** “coarse HGB,” “future-label exclusion,”
  “all fifteen predeclared tests,” “post-review,” “referee2-final,” “for publication
  and deployment,” “previously similar rounded values,” and any assertion that
  non-significance proves no harm.

## Lean appendix plan

Keep the appendix useful to a reviewer without turning it into code documentation.

- **Appendix A -- Data roles and reproducibility.** Compact run/scene partition,
  sampling seed, causal resampling rule, hardware/software, release tag/manifest,
  and one-shot receipt. Detailed checksums and per-forecast files belong in the
  repository, not a large paper table.
- **Appendix B -- Robustness.** One compact table for ensemble size, time window,
  guard shape, axis, and any completed with-walls/training-recipe sensitivity.
- **Appendix C -- Statistical definitions.** Crossed run/draw resampling,
  equivalence margins, exact tail definitions, and the unattainable oracle. Do not
  reproduce every run and every p value.

The notation table should remain in Method (`tab:notation`). Remove full code
listings, internal filenames, and implementation walk-throughs from the appendix.
Every appendix section must be cited in the main text with `\appref`.

## Response-letter synchronisation

The response skeleton is methodologically sound but stale in several places.

1. Replace its opening claim that `main.tex` integration is pending: the front half
   is now integrated, while results remain pending.
2. Update M8/M9 implementation status only after planner/separation artifacts and
   saved logs exist; the current text still says the planner is unimplemented.
3. Resolve every square-bracket placeholder from signed artifacts and insert final
   page/line ranges only after PDF compilation.
4. For each M1--M11 reply, lead with the concrete change, then the result, then the
   manuscript location. Avoid repeating “we agree” without evidence.
5. Record the corrected pre-held-out tag/commit, test log, output checksums, and
   one-shot execution receipt in the verification block.
6. Never copy reviewer-oriented phrases (“the referee-3 pipeline,” “post-review,”
   branch names) into the manuscript. They belong only in the response letter.

## Submission declarations and package checks

- Add CRediT, funding, and an applicable generative-AI declaration after the
  competing-interest statement.
- Update Data availability with the T-junction DOI and exact derived-output archive.
- Update Code availability only after a neutral public tag and persistent archive
  exist; name the licence, environment, and regeneration entry point.
- Supply 3--5 highlights, each <=85 characters, outside `main.tex` if required by
  the submission system.
- Produce a simplified graphical abstract from Figure 2 only if time permits; it is
  optional and must not delay the evidentiary revision.
- Before submission: no placeholders, review-history strings, unresolved citations,
  broken hyperlinks, appendix-prefix duplication, obsolete parameter counts, or
  result numbers that lack a signed artifact.

## Final integration sequence

1. Freeze/verify code and run artifacts; do not write results from console output.
2. Generate the signed result manifest and decide each claim with the gate table.
3. Replace lines 228--334 and rebuild the lean appendices.
4. Regenerate Figures 1--4 and final tables from the same signed artifacts.
5. Add declarations, archive identifiers, and release metadata.
6. Fill the response letter with exact result and manuscript references.
7. Compile, render every PDF page, visually inspect figures/tables/fonts/hyperlinks,
   and run a final search for all removal-manifest terms.

