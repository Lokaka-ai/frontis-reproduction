# Checkpoint 3 — Short Baseline Trajectory

> **Audit note (2026-08-16):** The trajectory remains descriptively valid, but the saved checkpoint does not persist exact island membership. Built-in journal-based restoration is not population-faithful for counterfactual policy resumption. See [the complete research-design reevaluation](../../analysis/RESEARCH_DESIGN_REEVALUATION.md).

Status: complete. Checkpoint 4 was not started.

## Run identity

- Task: `spooky-author-identification`
- Run: `checkpoint3-baseline`
- Output: `/workspace/frontis-research/pilot/outputs/checkpoint3-baseline`
- OpenRSI commit: `ece6cbdf115ed72c3b62643a836504d77365e3a0`
- MLE-Bench commit: `507f92e1138bb6e40dac5c6ee7a6758e6424bf97`
- Search: 5 configured generations, 2 individuals/generation, 11 maximum steps
- Operators: 2 Draft, 3 Improve, 3 Crossover, 2 Debug
- Parent utility weights: score 1.0, delta 0.4, novelty 0.25
- Limits: 1,800 s search, 2,400 s model+sandbox, 600 s sandbox job

## Outcome

- Generated nodes: 10 (plus the synthetic root)
- Valid / invalid: 8 / 2
- Hidden validation score progression:
  `0.36072, 0.47847, 0.49220, 0.42358, 0.46160, 0.41527, null, 0.36072, null, 0.45213`
- Best node: `8f743f54ce664e04aadaa790fbb0c6ae` (Draft), score `0.36072`
- Final official submit score: `0.36045`
- Candidate tokens: 51,526 prompt; 54,234 completion; 105,760 total
- Candidate runtimes: 468.149 s model; 1,000.574 s sandbox; 1,468.723 s combined

## Trajectory behavior

- Parent selection repeatedly favored the best Draft with utility 1.1768 and probability
  0.7311, but stochastic sampling also selected the weaker Draft (probability 0.2689)
  for two Improve children. Crossovers combined the two Draft branches or later combined
  a Draft with the first Crossover.
- The best score was established by the first Draft and was never beaten. One Debug repair
  exactly recovered `0.36072`; the other repaired successfully but scored `0.45213`.
- The two invalid nodes were compute-heavy Crossovers. Step 6 stopped after partial
  five-fold hyperparameter tuning at 92.236 s. Step 8 stopped after four folds at the
  600.364 s sandbox limit. Neither card stored an `error_signature`, so the normalized
  table leaves that field null and retains the feedback/log paths as evidence.
- Novelty started at 1.0, fell to 0.7071, reset to 1.0 when the new `ensemble` family
  appeared, then declined monotonically to 0.3333. The final family failure rates stored
  by the strategy board are 0.2222 for `sklearn` and 0.0 for `ensemble`.
- Delta behavior is consistent with a minimized metric: positive deltas correspond to
  lower child scores, while negative deltas correspond to regressions. Debug nodes and
  Drafts have null deltas in the source cards.

## Normalized artifacts

- `trajectory_extract/trajectory.csv`
- `trajectory_extract/trajectory.jsonl`
- `trajectory_extract/trajectory_summary.json`
- Extractor: `scripts/extract_trajectory.py`

The normalized table includes every required handoff field and also retains the paths to
reasoning, feedback, clear/raw logs, and experience cards. Missing source values remain
null; the extractor does not manufacture error signatures or Debug deltas.

## Verification

- Extractor syntax compilation: passed
- Extractor lineage, required-column, duplicate-node, and retained-path checks: passed
- Model server health: `{"status":"ok"}`
- Sandbox health: `{"status":"ok","sandbox":"chroot-seccomp","task":"spooky-author-identification"}`
- SSH issues: none
