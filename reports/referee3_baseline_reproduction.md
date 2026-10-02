# Referee-3 baseline reproduction

Date: 2 October 2026 (Australia/Sydney)

## Frozen starting point

- Baseline commit: `73b945b` (`origin/referee2-final`)
- Handoff commit: `fd8f827`
- Working branch: `trc-independent-review`
- Frozen numerical namespace: `data/processed/referee2_*`
- New work namespace: `referee3_*`

The frozen outputs were not overwritten.  The Desktop worktree is the only
checkout used for this revision; the author's source checkout is read-only and
is used only for its existing Python environment and raw-data links.

## Test reproduction

The unmodified baseline suite was run with the source checkout's virtual
environment and this worktree's `src` directory on `PYTHONPATH`:

```powershell
$env:PYTHONPATH='C:\Users\61411\Desktop\GEL-Ped-TRC-Review\src'
& 'C:\Users\61411\Documents\Codex\2026-07-22\re\work\pedestrian_geometry\.venv\Scripts\python.exe' -m pytest -q
```

Result: **38 passed**.  Six expected scikit-learn warnings were emitted by
short synthetic MLP tests (three batch-size clips and three non-convergence
warnings at eight test epochs); there were no failures.

## Environment

- Operating system: Windows
- Python environment: project virtual environment from the frozen source
  checkout
- PyTorch: 2.14.1+cpu
- Available GPU: NVIDIA GeForce RTX 3050, 6 GB; the installed PyTorch build is
  CPU-only, so graph training in this revision is CPU-bound
- GPU driver: 591.86

Exact package versions are recorded in `requirements-lock.txt` and
`requirements-graph.txt`.  Hardware timing is recorded separately after the
referee-3 implementation is frozen.

## Audit findings before modification

Three correctness/fairness issues were confirmed before code changes:

1. five retained rows did not represent a fixed causal time window after the
   random 6,000-row cap;
2. the primary evaluation excluded cases using future target speed; and
3. ETH/UCY used interaction decay 0.35 m while the manuscript and controlled
   pipeline specified 0.8 m.

The reviewer-3 pipeline therefore uses full chronological inference origins,
a time-indexed guard, a past-only quality screen, and one interaction decay
specified before the new held-out test.
