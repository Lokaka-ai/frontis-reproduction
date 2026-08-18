# Frontis selector activation: confirmatory lab brief

Date: 2026-08-16
Checkpoint status: **met; stop before a downstream-value experiment**

## Evidence-grounded claim

In three fresh, independent short trajectories on `spooky-author-identification`
using Frontis-MA1-35B-Q4 and the audited OpenRSI/OpenMLE-Evo search path, the
rule-based novelty component was non-discriminative in **all 8/8 eligible exact
live-population decision states, replicated in 3/3 trajectories**. Every legal
candidate in those states was assigned the same coarse method family, `sklearn`,
so novelty added an equal constant to every candidate utility and changed no
selection probability.

This is evidence of a concrete controller activation failure in this setting:
the configured novelty weight was nonzero (`0.25`), but the novelty mechanism
exerted zero selection pressure. It is **not** evidence that novelty is generally
useless, that Frontis search performance was harmed, or that a replacement
selector would improve downstream outcomes.

## Preregistered primary result

The unit is a unique trajectory × generation × exact live-island population, not
an individual draw. Repeated draws from the same generation state were retained
for audit and deduplicated for the primary counts.

| Root seed | Eligible legal states | Score discriminative | Delta discriminative | Novelty discriminative | Retained child cards | Failed child cards |
|---:|---:|---:|---:|---:|---:|---:|
| 20260816 | 3 | 3 | 2 | 0 | 8 | 1 |
| 20260817 | 2 | 2 | 1 | 0 | 8 | 3 |
| 20260818 | 3 | 3 | 2 | 0 | 8 | 0 |
| **Total** | **8** | **8 (100%)** | **5 (62.5%)** | **0 (0%)** | **24** | **4** |

The same qualitative pattern occurred in every trajectory: score discriminated
at every eligible state; delta was tied in the first eligible state and became
discriminative later; novelty remained tied throughout. The earlier seed-42
baseline showed the same novelty inactivity but was discovery evidence and was
not pooled into these counts.

## Post-hoc mechanism and magnitude audit

These checks explain the primary observation; they are labeled post hoc and do
not change the preregistered endpoint.

- All 8/8 legal states were single-family states, and the only observed legal
  family was `sklearn`.
- The implementation detects a small rule-based set of families from imports and
  code tokens. Candidate novelty is inverse-square-root family frequency. When
  every legal candidate has the same family, every candidate receives the same
  novelty value; a common utility offset cancels exactly under softmax.
- There were 6 eligible one-parent `Improve` states. Full utility equaled score-only
  probabilities exactly in 3/6. In the other 3/6, delta changed the distribution
  with total-variation distances `0.0733`, `0.0778`, and `0.0996`.
- Delta never changed the argmax parent set in any of the 6 Improve states. Thus,
  within this sample, score determined the top-ranked parent, while delta sometimes
  made a moderate change to stochastic sampling probabilities.
- Crossover states were included in component-activation counts but excluded from
  the one-parent probability comparison because crossover samples unordered pairs.

## Why the result is legally and logically interpretable

- Candidate support came from exact serialized live-island membership, never the
  archive/journal. Trace candidate order had to match the preceding serialized
  population exactly.
- The common selection metric was a fixed hidden validation holdout inaccessible
  to generated programs. Program-reported validation scores and official/private
  test scores were not substituted for this metric.
- Utility and softmax probabilities were independently recomputed from stored
  components, fixed weights, and temperature to `1e-10` tolerance.
- Exact prompt messages, completions, request kwargs, and deterministic
  call-specific seeds were present for all **48 traced LLM calls**. Per-operator
  request seeds were unique within each trajectory.
- All 24 child cards and 4 failed children were retained. Each trajectory had nine
  journal nodes and `runner_failures=[]`.
- Three independent root seeds were fixed before valid confirmatory generation.
  Outcomes were not inspected between trajectories.

## Deviations and exclusions

- A fixed-seed attempt was stopped and excluded before outcome analysis because
  identical Draft prompts with the same request seed produced duplicate samples.
- A second attempt was stopped and excluded because a pod restart removed the
  root-only hidden-answer file, causing the scorer to refuse scores. The exact
  holdout was restored only after matching persistent IDs, order, text, and counts.
- Seed 20260816 completed its fixed search before an unintended post-search
  self-validation evaluation ran. That result is not official, is excluded from
  every endpoint, and could not affect already-recorded search states. A tested,
  benchmark-agnostic `final_submit=false` gate prevented this for seeds 20260817
  and 20260818.
- The first seed-20260817 launch failed in Hydra before creating an output directory
  or making a model call because the new config key required append syntax. It is
  not a trajectory.
- An analyzer initially rejected two generation-3 checkpoints in seed 20260816.
  Inspection showed boundary and terminal snapshots with canonically identical
  complete population payloads but different step counters. The analyzer now
  accepts multiple snapshots only when their full population payloads are
  identical; non-equivalent duplicates remain a hard failure.

## Interpretation for the lab meeting

The defensible takeaway is not “Frontis chooses bad parents.” It is:

> On this inexpensive text-ML task, the audited Frontis/OpenMLE-Evo controller's
> intended novelty pressure collapsed completely because a coarse rule-based family
> label mapped every legal candidate to `sklearn`. Score remained active everywhere;
> delta sometimes shifted sampling probabilities but did not change the top-ranked
> parent in these trajectories.

This identifies a plausible design bottleneck worth testing: diversity signals
defined at too coarse a method-family level can be configured yet operationally
inactive in homogeneous populations. The next scientific step should test whether
this repeats on another inexpensive task or whether a predeclared finer code/behavior
diversity measure activates without degrading search. No downstream-performance or
causal-improvement claim should be made until such an intervention is evaluated with
legal-action support and fixed budgets.

## Evidence artifacts

- Frozen design: `experiments/selector_component_confirmation/PREREGISTRATION.md`
- Primary analyzer: `analysis/analyze_legal_selector_components.py`
- Primary frozen output: `selector-component-confirmation-analysis/`
- Extended mechanism output: `selector-component-confirmation-analysis-v2/`
- Valid trajectories: `selector-component-confirmation/seed-20260816`,
  `seed-20260817`, and `seed-20260818`
- Design reevaluation: `analysis/RESEARCH_DESIGN_REEVALUATION.md`
