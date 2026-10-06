# Submission positioning and release notes

Author: Amir Ghorbani

## Positioning

The strongest outlet fit is a transport methods journal interested in
operational prediction, trustworthy learning, and deployable decision support.
The manuscript itself should not name or address a target journal.

The central value proposition is narrow and testable: GEL-Ped leaves a modern
short-horizon graph predictor fully active while independently trained members
agree within a complete-run calibration envelope, then responds immediately to
an observable instability by moving smoothly toward a bounded feature-based
fallback.

## Claims to preserve

- The deployed score and action are causal and local to one pedestrian.
- Complete runs or scenes, never shuffled frames, define evaluation units.
- The fallback and matched controls use the same information budget specified
  by the frozen protocol.
- Accuracy, tail risk, detection, retention, intervention rate, retraining
  sensitivity, and planner consequences answer different questions and should
  not be collapsed into one headline metric.
- Member disagreement detects variance-type instability but is blind to
  errors shared by the ensemble.
- The planner replay is open-loop with recorded pedestrians; it is not a
  reciprocal simulation or safety proof.

## Claims to avoid

- universal robustness to distribution shift;
- a formal uncertainty-coverage or collision-safety guarantee;
- state-of-the-art long-horizon trajectory prediction;
- causal attribution of every gain to one behavioural feature;
- describing development-informed crossing data as untouched confirmation;
- implying that the current referee-3 driver automatically reruns Tordeux or
  ETH/UCY experiments.

## Submission package checklist

- verified manuscript PDF and editable Overleaf source;
- bibliography and only the figures used by the manuscript;
- response-to-referee document with every placeholder resolved;
- public code at the exact release tag cited in the manuscript;
- `CITATION.cff`, MIT code licence, data-provenance statement, and environment
  lock files;
- compact result tables/manifests required to audit manuscript values;
- no raw third-party trajectories, caches, auxiliary LaTeX files, secrets,
  temporary renders, or workstation-specific paths;
- archived one-shot prediction receipt and confirmatory outputs, including
  null or adverse findings;
- a permanent software archive DOI added when available, without delaying the
  reproducible Git tag.
