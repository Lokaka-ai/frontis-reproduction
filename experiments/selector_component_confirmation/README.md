# Selector-component confirmation

This directory contains the frozen protocol and small non-sensitive summaries for
the valid confirmation experiment.

## Design

- Task: `spooky-author-identification`
- Model: `Frontis-MA1-35B-Q4_K_M.gguf`
- Root seeds: `20260816`, `20260817`, `20260818`
- Primary unit: unique trajectory × generation × exact live-population state
- Primary estimand: whether each utility component has nonzero range across the
  exact legal candidates
- Utility weights: score `1.0`, positive delta `0.4`, novelty `0.25`

Read `PREREGISTRATION.md` before interpreting the results. The primary summary is
`results/analysis_summary.json`; `results/legal_state_components.csv` contains one
row per eligible legal state. Per-seed runner summaries are included for provenance.

## Result boundary

The data establish that novelty was non-discriminative at 8/8 eligible states in
these three short trajectories. They do not estimate a general task distribution,
causal downstream search value, or the performance of a replacement selector.

Raw prompts, completions, generated code, checkpoints, hidden labels, and model
weights are excluded from Git because they are large, sensitive, or externally
licensed. They are required for a byte-level rerun but not for checking the committed
summary calculations.
