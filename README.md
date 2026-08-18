# Frontis-MA1 / OpenMLE-Evo Search-Controller Study

This repository contains a bounded academic reproduction and mechanism audit of
Frontis-MA1 / OpenMLE-Evo, based on the upstream
[FrontisAI/OpenRSI](https://github.com/FrontisAI/OpenRSI) project.

## Scope

The present study asks whether the fixed score, progress, and novelty components
used by the evolutionary controller are behaviorally active at authentic parent
decision states. It uses one inexpensive MLE-Bench task,
`spooky-author-identification`, rather than claiming full benchmark reproduction.

The main result is intentionally narrow:

> Across three fresh short trajectories, method-family novelty was
> non-discriminative at all 8/8 eligible exact live-population states. Every legal
> candidate was labeled `sklearn`, so novelty added the same constant to all parent
> utilities and had no effect on selection probabilities in those states.

This does **not** establish that novelty harms downstream performance, that Frontis
generally selects poor parents, or that a replacement selector would improve search.

Start with the [complete report](analysis/FRONTIS_RESEARCH_COMPLETE_MEETING_REPORT.md),
then read the [preregistration](experiments/selector_component_confirmation/PREREGISTRATION.md)
and [design reevaluation](analysis/RESEARCH_DESIGN_REEVALUATION.md).

## Repository organization

- `analysis/`: analysis programs, final report, and validity audit.
- `experiments/selector_component_confirmation/`: frozen protocol and small
  result summaries for the valid confirmation.
- `experiments/exploratory_baseline/`: explicitly exploratory historical record.
- `scripts/`: trajectory extraction, counterfactual probe, validation restoration,
  and confirmation launcher.
- `third_party/openrsi_patches/`: cumulative and historical patches against the
  pinned OpenRSI checkout.
- `AGENTS.md`: the evidence and reporting rules governing this project.

## Reproduce the confirmation

1. Clone OpenRSI at commit `ece6cbdf115ed72c3b62643a836504d77365e3a0`.
2. From the root of the `OpenRSI/` checkout, apply
   `third_party/openrsi_patches/0004_final_validity_instrumentation.patch`.
   The patch paths deliberately begin with `OpenMLE-Evo/`, so use an absolute path
   to this research repository's patch file if the repositories are separate.
3. Prepare the Frontis-MA1 model, MLE-Bench task assets, llama.cpp endpoint, and
   isolated scorer as described in the report. Model weights and datasets are not
   redistributed here.
4. Set `FRONTIS_RESEARCH_ROOT` to the directory containing `OpenRSI/`, then run:

   ```bash
   scripts/run_selector_component_confirmation.sh 20260816
   scripts/run_selector_component_confirmation.sh 20260817
   scripts/run_selector_component_confirmation.sh 20260818
   ```

5. Analyze the three output directories:

   ```bash
   python analysis/analyze_legal_selector_components.py \
     --run 20260816=/path/to/seed-20260816 \
     --run 20260817=/path/to/seed-20260817 \
     --run 20260818=/path/to/seed-20260818 \
     --output-dir /path/to/analysis-output
   ```

Run `python analysis/analyze_legal_selector_components.py --help` for the exact CLI.
The committed `results/` summaries allow the reported counts to be inspected
without model weights or private infrastructure.

## Dependencies

The legal-state analyzer uses only the Python standard library. The exploratory
analysis programs additionally require the packages in `requirements-analysis.txt`.
The generation environment follows the pinned OpenRSI dependencies.

## Reproducibility conventions

Experiment configurations, scripts, and small result summaries should be committed. Experiment records should identify the model and version, random seed, task set, relevant budgets, and exact upstream OpenRSI commit.

Large datasets, model checkpoints, raw prompts/completions, generated code, logs,
credentials, and hidden evaluation labels remain outside Git. The repository keeps
only executable research code, frozen design records, cumulative source patches,
and small non-sensitive summaries.
