# GEL-Ped development handoff

Author: Amir Ghorbani
Updated: 2 October 2026

## Objective

Maintain a technically consistent, readable, reproducible submission package
for *GEL-Ped: An instability-triggered safeguard for short-horizon pedestrian
prediction under changing crowd flows*. Act as a critical scientific reviewer
before acting as a revising author. Do not fabricate results, tune on held-out
outcomes, or add model components only to improve a table.

## Git and scientific freezes

- Development branch: `trc-independent-review`
- Frozen referee-2 base: commit `73b945b`
- Leakage-safe referee-3 analysis freeze: commit `4f40ec4`
- Pre-held-out registry tag: `gelped-v1.0-preheldout`
- Frozen protocol: `configs/referee3_lock.json`
- Untouched archive registry: `configs/referee3_heldout_registry.json`

Always inspect `git status`, the two configuration hashes, and the registered
freeze before continuing scientific work. Do not overwrite the historical
`referee2_*` evidence. New outputs use the `referee3_*` namespace.

## Current method and scope

GEL-Ped predicts 0.4-s mean velocity from a causal 0.8-s history. Five
independently trained anchored graph predictors form the default forecast. A
pedestrian-local, time-indexed member-disagreement score is compared with an
out-of-fold complete-run calibration envelope. Inside the envelope the graph
weight is one; only an exceedance moves the forecast toward a feature-based
HGB residual fallback with an explicit speed cap.

The defensible claim is an auditable response to detectable variance-type
instability. Member disagreement cannot reveal an error shared by all members.
Do not market the method as a universal distribution-shift detector, conformal
guarantee, long-horizon generator, or safety certificate.

## Canonical files

- Manuscript: `manuscript/overleaf_trc/main.tex`
- Bibliography: `manuscript/overleaf_trc/references.bib`
- Release guide: `README.md`
- Reproduction protocol and equation map: `REPRODUCIBILITY.md`
- Data provenance: `data/README.md`
- Core generator: `experiments/referee3_generate_predictions.py`
- Main frozen analysis: `experiments/referee3_analyse.py`
- Backbone check: `experiments/referee3_transformer_generalization.py`
- Planner audit: `experiments/referee3_reactive_planner.py`
- One-shot prediction path: `experiments/referee3_generate_heldout.py`
- One-shot analysis: `experiments/referee3_analyse_heldout.py`
- Referee response draft: `reports/response_to_referee_round3_draft.md`

The current referee-3 controlled-data driver covers the Jülich
corridor/crossing batches. The registered T-junction has a separate one-shot
path. Tordeux and ETH/UCY adapters and historical workflows are retained but
are not silently executed by the current driver.

## Non-negotiable held-out firewall

Do not inspect T-junction errors until all method, inclusion, endpoint, and
analysis choices are committed and tagged. First run inventory-only validation.
Then generate target-blind predictions exactly once, archive their receipt and
fitted caches, and only then run the frozen confirmatory analysis into a new,
immutable directory. Report null and adverse results as produced. Full commands
and checks are in `REPRODUCIBILITY.md`.

## Release completion checks

1. Run the complete test suite, Ruff, `git diff --check`, and `pip check`.
2. Confirm code, configuration, tables, figures, prose, and response letter use
   the same task definition and model name.
3. Compile `manuscript/overleaf_trc/main.tex`, inspect every rendered page, and
   update `README_OVERLEAF.txt` with the verified page count.
4. Package only manuscript source, bibliography, required figures, and a clean
   PDF for Overleaf; exclude auxiliary files and raw data.
5. Keep raw third-party data, caches, secrets, temporary renders, and local
   absolute paths out of the GitHub release.
6. Create the final version tag only after the one-shot evidence and manuscript
   are frozen together.
