# Response to reviewer

We thank the reviewer for a technically precise report. The revision changes
both the method and the paper's central claim. GEL-Ped is now a
geometry-guarded graph learner rather than a prospective-feature model whose
transfer claim rests mainly on anchoring.

## Major comments

1. **What drives transfer.** We changed the title, abstract, introduction and
   conclusion. The new claim is geometry guarding, not that the prospective
   feature vector alone solves topology shift. A new 2 x 2 experiment shows
   that prospective features help both learner families in the corridor sets
   but not the standalone boosted learner in crossing. This boundary is stated
   explicitly.

2. **Router behaviour.** We now report realised gate distributions by run.
   Mean graph weight is 0.690 and 0.691 in the two corridor tests and 0.397 in
   crossing. New diagnostics include g=1, g=0, the calibration-best fixed
   blend, a label-using oracle fixed blend, a same-learner 2 x 2 experiment,
   and buffer-level disagreement versus relative expert error.

3. **Cross-fitting and buffer definition.** Every calibration disagreement is
   produced by leave-one-complete-run-out refitting. The threshold is a
   finite-sample 90% quantile of out-of-fold frame scores. The online buffer is
   the current and previous four sampled frames (at most 2.0 s), never a
   whole-run mean. Threshold sensitivity covers 80%, 90%, 95%, and Tukey
   rules.

4. **Development chronology.** The paper now states that crossing was inspected
   in earlier exploration and is a structured stress test rather than pristine
   external confirmation. An official naturalistic ETH train/test split was
   added after the guarded architecture and protocol were frozen.

5. **Essential baselines.** The main table now includes constant velocity,
   calibrated anisotropic Social Force, time-to-collision, Anticipation
   Velocity Model, direct MLP, and a graph interaction network under the same
   0.8-s-history/0.4-s-target protocol.

6. **Inference.** Five- and three-run corridor findings are described as
   descriptive. Paired run-bootstrap intervals, exact sign-flip tests, and Holm
   correction are reported. Shared sessions and participants are stated as a
   dependence limitation.

7. **Practical magnitude and noise.** The revision states that raw tracks are
   unsmoothed and adds a coordinate-smoothing audit. It translates the main
   crossing difference to 0.4-s displacement and reports a 0.8-m collision
   screen with false-alarm and miss rates.

8. **Hard cases.** Error is now stratified by projected conflict risk,
   closest-approach time, and clearance. The text states the measured result:
   the gain is present in high-risk cases but is not concentrated exclusively
   there.

9. **Naturalistic scope.** The held-out biwi_eth sequence is evaluated with
   past-motion route estimation and no cardinal destination. The graph network
   is slightly better than GEL-Ped on this split, which is reported as a
   boundary of the guard's value.

10. **Related work and learned baseline.** The review now connects GEL-Ped to
    Neural Social Physics, TrajNet++, constant-velocity analysis, deep
    ensembles, conformal prediction, and a recent higher-order graph model. A
    compact attention-pooling graph network with 13,395 parameters is trained
    on the exact study task.

11. **Multi-step claims.** The earlier rollout section and undefined tensor
    comparator were removed. The revised paper makes a one-step rolling
    prediction claim only.

12. **Wall input.** The wall-free specification is now the primary model and is
    stated in the method.

## Processing and reporting comments

The revision uses a 0.8-s velocity history; reports sensitivity to 0.4 s;
expands neighbours from 3 m/eight to 5 m/twelve in a frozen-model audit; and
includes every observed speed in a separate evaluation. Five-seed means are
the primary estimates. The key comparator has one name throughout: matched
direct expert system. Feature dimensions, attention, graph capacity, route
estimation, split identifiers and numerical settings are documented. The
appendix contains only reproducibility and attribution information; coding
details remain in the repository.
