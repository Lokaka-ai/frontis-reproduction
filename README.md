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

The commands below assume this repository is at
`/workspace/frontis-research`. Kaggle requires an accepted competition agreement and
credentials at `~/.kaggle/kaggle.json`. Dataset files and hidden labels remain outside
Git.

### 1. Prepare the pinned source checkouts

```bash
export FRONTIS_RESEARCH_ROOT=/workspace/frontis-research
export MLEBENCH_ROOT=/workspace/mle-bench
export MLEBENCH_DATA_DIR=/workspace/mle-bench-data

git clone https://github.com/FrontisAI/OpenRSI.git \
  "${FRONTIS_RESEARCH_ROOT}/OpenRSI"
git -C "${FRONTIS_RESEARCH_ROOT}/OpenRSI" checkout \
  ece6cbdf115ed72c3b62643a836504d77365e3a0
git -C "${FRONTIS_RESEARCH_ROOT}/OpenRSI" apply \
  "${FRONTIS_RESEARCH_ROOT}/third_party/openrsi_patches/0004_final_validity_instrumentation.patch"

git clone https://github.com/openai/mle-bench.git "${MLEBENCH_ROOT}"
git -C "${MLEBENCH_ROOT}" checkout \
  507f92e1138bb6e40dac5c6ee7a6758e6424bf97
git lfs install
git -C "${MLEBENCH_ROOT}" lfs pull
```

The patch paths begin with `OpenMLE-Evo/`, so apply the patch from the root of the
OpenRSI checkout as shown above.

### 2. Download MLE-Bench data and create the pilot assets

```bash
python3.12 -m venv "${FRONTIS_RESEARCH_ROOT}/.venv-assets"
source "${FRONTIS_RESEARCH_ROOT}/.venv-assets/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "${MLEBENCH_ROOT}"
python -m pip install -r "${FRONTIS_RESEARCH_ROOT}/requirements-assets.txt"

mlebench prepare \
  -c spooky-author-identification \
  --data-dir "${MLEBENCH_DATA_DIR}"

python "${FRONTIS_RESEARCH_ROOT}/scripts/prepare_spooky_assets.py" \
  --mlebench-data-dir "${MLEBENCH_DATA_DIR}" \
  --assets-root "${FRONTIS_RESEARCH_ROOT}/pilot/assets"
```

The preparation script verifies the pinned MLE-Bench source, then creates and checks:

```text
pilot/assets/
├── leaderboards/spooky-author-identification.csv
├── manifest/spooky-author-identification.parquet
├── submit/spooky-author-identification/train.csv
└── validation/spooky-author-identification/
    ├── sample_submission.csv
    ├── test.csv
    └── train.csv
```

The source has 17,621 rows. The fixed stratified 80/20 split uses
`random_state=42` and produces 14,096 visible training rows and 3,525 hidden rows.
`validation/test.csv` contains only `id` and `text`.

### 3. Restore the scorer-only answers

Run this only on the host. Do not mount the secure directory into the generated-code
sandbox.

```bash
sudo "${FRONTIS_RESEARCH_ROOT}/.venv-assets/bin/python" \
  "${FRONTIS_RESEARCH_ROOT}/scripts/restore_validation_secure.py" \
  --assets-root "${FRONTIS_RESEARCH_ROOT}/pilot/assets" \
  --secure-root /var/lib/frontis-pilot/secure/validation
```

The sandbox service must mount these public inputs read-only:

```text
/workspace/frontis-research/pilot/assets/validation -> /datasets/validation
/workspace/frontis-research/pilot/assets/submit     -> /datasets/submit
```

Set these OpenMLE-Evo `.env` values to the host asset paths and sandbox submit path:

```bash
OPENMLE_EVAL_DATA=/workspace/frontis-research/pilot/assets/manifest/spooky-author-identification.parquet
OPENMLE_LEADERBOARD_DIR=/workspace/frontis-research/pilot/assets/leaderboards
OPENMLE_SUBMIT_DATA_DIR_ROOT=/datasets/submit
```

Before generation, verify that the model endpoint and isolated scorer are healthy and
that generated programs cannot read
`/var/lib/frontis-pilot/secure/validation`. Model weights, service credentials, and
hidden labels are not included in this repository.

### 4. Run the three confirmation trajectories

With the Frontis-MA1 llama.cpp endpoint and sandbox scorer running:

```bash
cd "${FRONTIS_RESEARCH_ROOT}"
scripts/run_selector_component_confirmation.sh 20260816
scripts/run_selector_component_confirmation.sh 20260817
scripts/run_selector_component_confirmation.sh 20260818
```

### 5. Analyze the outputs

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

The asset scripts use the pinned packages in `requirements-assets.txt`. The legal-state
analyzer uses only the Python standard library. The exploratory analysis programs
additionally require the packages in `requirements-analysis.txt`. The generation
environment follows the pinned OpenRSI dependencies.

## Reproducibility conventions

Experiment configurations, scripts, and small result summaries should be committed. Experiment records should identify the model and version, random seed, task set, relevant budgets, and exact upstream OpenRSI commit.

Large datasets, model checkpoints, raw prompts/completions, generated code, logs,
credentials, and hidden evaluation labels remain outside Git. The repository keeps
only executable research code, frozen design records, cumulative source patches,
and small non-sensitive summaries.
