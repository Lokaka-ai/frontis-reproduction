# Frontis-MA1 / OpenMLE-Evo Reproduction

This repository supports an academic reproduction and evaluation of Frontis-MA1 / OpenMLE-Evo, developed in the upstream [FrontisAI/OpenRSI](https://github.com/FrontisAI/OpenRSI) project.

## Scope

The initial goal is to reproduce the published MLE-Bench experiments. Planned follow-up work includes:

- evaluating additional MLE-Bench and Kaggle tasks;
- comparing backbone language models; and
- analyzing failure modes to inform future research.

Setup and run instructions will be added as the reproduction pipeline is implemented.

## Repository organization

- `configs/`: version-controlled experiment configurations.
- `scripts/`: reproducible setup, execution, and evaluation entry points.
- `analysis/`: analysis code and small, reviewable result summaries.
- `experiments/`: experiment records and metadata.
- `third_party/`: notes about external dependencies and pinned upstream versions.

## Reproducibility conventions

Experiment configurations, scripts, and small result summaries should be committed. Experiment records should identify the model and version, random seed, task set, relevant budgets, and exact upstream OpenRSI commit.

Large datasets, model checkpoints, generated experiment artifacts, credentials, and other large or private assets must remain outside Git. Keep `main` clean and use branches and pull requests for collaborative work.
