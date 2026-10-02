# GEL-Ped: handoff to the next AI

Date: 2 October 2026 (Australia/Sydney)

## Mission

Continue an independent Transportation Research Part C-style audit and revision
of GEL-Ped. Work as a critical referee first and as the revising author second.
Do not defend the current manuscript by default, do not fabricate stronger
results, and do not stack methods merely to improve a table. The desired end
state is a technically consistent, readable, reproducible submission package
with a candid readiness recommendation.

## Workstation and Git state

- Dedicated workstation: `C:\Users\61411\Desktop\GEL-Ped-TRC-Review`
- Current branch: `trc-independent-review`
- Upstream frozen branch: `origin/referee2-final`
- Starting commit: `73b945b`
- Author's source checkout (do not edit):
  `C:\Users\61411\Documents\Codex\2026-07-22\re\work\pedestrian_geometry`
- GitHub frozen snapshot:
  `https://github.com/AmirNetwork/GEL-Ped/tree/referee2-final`
- Previous AI task/thread:
  `01a0fa86-f882-7a63-8bf8-98bf22169cf1`

The Desktop worktree is currently at the frozen commit. The only untracked item
at handoff is `work_initial-01.png`, a rendered first-page QA image. Do not edit
or clean the author's source checkout. Raw datasets are intentionally untracked;
read them from the source checkout or use safe links if experiments require
them.

## Work already completed

- The final manuscript, code, processed outputs, response letter, figures, and
  Overleaf package were assembled at commit `73b945b`.
- The baseline test suite passes: **38 tests passed**.
- The 13-page PDF previously compiled without undefined references, missing
  citations, overfull boxes, or LaTeX errors.
- A demanding external referee report was addressed in the current version,
  including complete-run splits, all 15 Holm tests, a five-member graph
  ensemble, coarse-HGB and robust-aggregation controls, per-run crossing
  values, five ETH/UCY scenes, additional bottleneck/unidirectional runs,
  collision counts, route-axis and all-speed audits, and recursive horizons.
- No methodological revision has yet been made in this Desktop worktree.

## Primary files

- Manuscript source: `manuscript/overleaf_trc/main.tex`
- Current PDF: `manuscript/overleaf_trc/main.pdf`
- Bibliography: `manuscript/overleaf_trc/references.bib`
- Current reviewer response: `reports/response_to_referee_round2.md`
- Frozen method record: `configs/referee2_lock.json`
- Main experiment: `experiments/referee2_revision.py`
- Statistical family: `experiments/referee2_statistics.py`
- Operational/rollout audit: `experiments/referee2_operational_audit.py`
- Route and label-filter audits: `experiments/referee2_route_axis.py`
- Capacity sweep: `experiments/graph_capacity_sweep.py`
- Figure generation: `experiments/referee2_figures.py`
- Core implementation: `src/pedgeom/`
- Final numerical outputs: `data/processed/referee2_*`

External evidence and background:

- Latest referee report:
  `C:\Users\61411\.codex\attachments\e505c5d5-5cd6-4d46-9623-1a42df8b4b6d\Pasted text.txt`
- Earlier major report:
  `C:\Users\61411\.codex\attachments\0c7866f6-8529-43ba-9ee8-e69c6b3d0171\Pasted text.txt`
- Original 2022 paper:
  `C:\Users\61411\Downloads\2211.10792v1.pdf`

## Current method

GEL-Ped means Guarded Ensemble Learning for Pedestrians. It predicts mean
velocity over the next 0.4 s from a nominal 0.8-s history. Five independently
initialized graph residual learners (width 48, 25 epochs, 13,395 parameters
each) form the default graph mean. Their per-agent prediction spread is smoothed
over at most five retained forecast origins. A leave-one-complete-run-out
envelope gives `tau = 0.03791 m/s` and `S = 0.01176 m/s`. The graph weight is

`g_i = exp[-(E_i - tau)_+ / S]`,

so it is exactly one inside the envelope. Outside it, the prediction moves
toward an anchored, bounded coarse HGB residual based on current
behaviour-aligned fields. Prospective closest-approach geometry is a control,
not part of the primary model.

## Current evidence and permitted claim

- Familiar Juelich: graph 0.15964, GEL 0.15954 m/s.
- Altered geometry: graph 0.15299, GEL 0.15307.
- Crossing: graph 0.35427, GEL 0.34627; 13/13 runs favour GEL; mean difference
  -0.00801 m/s; median -0.00224; Holm-adjusted exact p = 0.003662.
- Run `d_1` is influential because one graph member reaches 0.8396 m/s. Without
  `d_1`, the mean reduction is about 1.3%.
- Highest-instability crossing stratum (>2 calibration IQR, n=2,875): graph
  0.56536, GEL 0.46653, about 17.5% lower.
- Coarse HGB is marginally lower in crossing (0.34553) but 5--8% worse in the
  other four controlled regimes.
- Bottleneck: graph/GEL/coarse = 0.24703/0.24722/0.26308.
- Unidirectional: 0.23957/0.23927/0.25986.
- Five ETH/UCY LOSO scenes: graph macro 0.226669, GEL 0.226225; wins 2/5;
  exact p = 0.9375. This is neutral external evidence.
- Recursive crossing displacement graph/GEL is 0.2591/0.2485 m at 0.8 s,
  0.6458/0.5888 at 1.6 s, and 1.9003/1.5593 at 3.2 s. Coarse HGB and constant
  velocity are better at 3.2 s; treat this only as a compounding-error audit.

The defensible claim is narrow: GEL-Ped preserves the graph ensemble in
supported conditions and suppresses a detectable neural-instability tail. It
does not universally improve accuracy, it is not a long-horizon trajectory
generator, and prospective geometry does not explain its main result. Crossing
informed development and remains a development-informed stress test; the five
ETH/UCY scenes are the cleaner external tests.

## Newly discovered issues that must be resolved first

### 1. Feature decay mismatch between manuscript/Juelich and ETH/UCY

The manuscript specifies interaction decay length `lambda = 0.8 m`
(`main.tex`, around line 87), and Juelich uses
`config["interaction_range_m"] = 0.8`. However,
`src/pedgeom/external.py` calls `_interaction_features` with
`interaction_range=0.35` (around line 119). Therefore the claim that the same
representation/protocol is reconstructed on ETH/UCY is not literally true.

Required action: determine whether 0.35 was an accidental default, an earlier
frozen ETH choice, or a justified scene-specific scaling. Do not silently pick
the better test result. Prefer a frozen consistency audit comparing 0.35 and
0.8 on calibration/validation rules. Either standardize before test evaluation
or explicitly disclose and justify the dataset-specific value. Update config,
code, outputs, manuscript, and response consistently.

### 2. The claimed two-second guard memory may be false after subsampling

The paper says a five-observation rolling score has a maximum memory of 2.0 s.
The code smooths over five retained rows per pedestrian, but controlled runs
are capped/subsampled to at most 6,000 forecasts per run. Retained origins may
not be consecutive 0.4-s steps and may have irregular frame gaps. Five retained
observations therefore do not automatically imply a two-second window.

Required action: trace sample selection and actual per-pedestrian frame gaps in
all regimes. Report the realized temporal-span distribution. Then either make
the guard explicitly time-based (recommended if possible), preserve all causal
origins for the guard before evaluation subsampling, or replace “maximum 2.0 s”
with an accurate statement. Add tests for irregular sampling and scene changes.

### 3. Learned/OOD guard controls do not use the same fallback

The primary instability guard blends the graph with **coarse HGB**. In
`experiments/referee2_revision.py`, the Mahalanobis and learned mixture-of-
experts controls blend the graph with the **prospective geometry HGB** (around
lines 511--514). This changes both the gate and the fallback, so those rows do
not isolate the gate signal and are not matched controls.

Required action: add genuinely matched controls in which instability,
Mahalanobis, supervised MoE, and graph--fallback disagreement all allocate the
same graph/coarse-HGB expert pair. Keep the older geometry-fallback variants
only if clearly labelled as separate ablations. Fit/tune gates using calibration
data only. Recompute the control table and revise any conclusions.

## Independent review questions

1. Is the contribution distinct enough from deep-ensemble uncertainty,
   selective prediction, uncertainty-triggered fallback, and classical mixture
   of experts?
2. Is the guard-to-fallback action principled, or mainly a post hoc response to
   one unstable run?
3. Are tail-risk endpoints rigorous and operationally meaningful beyond mean
   RMSE?
4. Are all controls genuinely matched in data, expert pair, training budget,
   seeds, and calibration?
5. Are the ETH/UCY adapter, frame-rate handling, graph features, and guard
   calibration identical to the claims in the manuscript?
6. Does any test-informed decision leak into a claimed confirmatory result?
7. Can the transport-systems value be explained for robots, AV--pedestrian
   interaction, and station monitoring without unsupported safety language?
8. Is the manuscript readable to transport researchers without specialist
   physics or deep-learning knowledge?

## Required workflow

1. Preserve commit `73b945b` as the frozen baseline.
2. Run the 38 tests before changes. The source repository's existing virtual
   environment may be used read-only; point `PYTHONPATH` to this worktree's
   `src` when necessary.
3. Write `reports/independent_trc_review.md` before implementing revisions.
4. Write `reports/revision_decision_log.md`. For every change, record whether it
   is accepted, rejected, or experimentally tested and whether calibration or
   test outcomes informed it.
5. Resolve the three consistency/fairness issues above before attempting any
   performance improvement.
6. If a new method is proposed, create a new config and output namespace. Do
   not overwrite the frozen `referee2_*` evidence.
7. Use complete-run inference, five neural seeds, exact run-level tests, and the
   full Holm family. Do not tune on crossing or held ETH/UCY scenes.
8. Update code, tests, processed outputs, manuscript, figures, README, cover
   letter, and response letter together.
9. Compile LaTeX, render and visually inspect every PDF page, check hyperlinks,
   citations, tables, figure fonts, and overflow.
10. Produce a clean Overleaf zip, final PDF, exact reproduction commands, and a
    candid recommendation: ready, ready only after specified evidence, or not
    ready.

## Writing constraints

- Keep the abstract conceptual; use only the few effect sizes needed to define
  the result and its limitation.
- Do not name the target journal inside the manuscript.
- Reduce jargon and motivate each component before equations.
- Connect literature directly to the gap and contribution.
- Do not over-discuss or criticize the unpublished 2022 predecessor.
- Keep the notation table and a compact, necessary appendix; exclude code-level
  clutter.
- Keep all figures professional, font-consistent, and hyperlinked.
- Do not hide the coarse-HGB crossing result, the influential `d_1` run, or the
  neutral ETH/UCY result.

## Definition of done

The handoff is complete only when the independent review exists, all material
accepted comments are implemented or explicitly declined with evidence, the
three newly identified consistency issues are resolved, tests and numerical
audits pass, the final PDF is visually clean, and the readiness recommendation
matches the evidence rather than the desired outlet.
