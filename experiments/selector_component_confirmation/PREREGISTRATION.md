# Frontis selector-component confirmation — preregistration

Status: **frozen before valid confirmatory generation; Amendments 1–3 applied**
Date frozen: 2026-08-16
Task: `spooky-author-identification`
OpenRSI commit: `ece6cbdf115ed72c3b62643a836504d77365e3a0` plus the research-observability changes captured cumulatively in `third_party/openrsi_patches/0004_final_validity_instrumentation.patch` (`0003` is a superseded intermediate export)
Model: `Frontis-MA1-35B-Q4_K_M.gguf`, llama.cpp, NVIDIA A40

## Research question

In short independent Frontis/OpenMLE-Evo trajectories on this task, do the score,
positive-delta, and method-family novelty components actually discriminate among
the candidates in the legal live population at authentic parent-decision states?

This is a controller-activation audit. It is not an estimate of downstream search
value and does not test whether any component is generally useful or harmful.

## Discovery/confirmation separation

The old seed-42 baseline is discovery evidence only. It suggested that novelty and
delta were constant over legal candidates. It will not be pooled into the primary
confirmation result.

The confirmation set comprises three fresh trajectories with root seeds
`20260816`, `20260817`, and `20260818`. Each operator uses a deterministic
call-specific request schedule `request_seed = root_seed + zero_based_call_index`.
The exact seed, prompt, and response are persisted for every call. A direct
llama.cpp check established identical output for an identical prompt/seed and
different output after changing the seed; this is an engineering check, not a
scientific observation.

### Amendment 1 — call-specific request seeds

The first attempted `20260816` run was stopped before completion and excluded
before any selector result analysis. The initial implementation sent the same
fixed request seed to every call. Identical Draft prompts therefore produced
identical samples and artificially removed within-run evolutionary diversity.
Its retained artifacts are stored under
`selector-component-confirmation-invalid-fixed-seed/`.

Before restarting any confirmation run, the seed schedule was corrected to use a
deterministic call-specific seed within each operator, and each operator's call
counter was added to the exact checkpoint. The run count, root seeds, budgets,
endpoints, and analysis plan were not changed. The analyzer rejects duplicate
request seeds within an operator and any request seed below its preregistered root.

### Amendment 2 — restart-safe validation scorer and no official submit

The first call-specific-seed attempt was also stopped and excluded before selector
analysis because the pod restart had cleared the root-only hidden validation answers.
Generated programs ran successfully, but the scorer correctly refused to assign a
score. Its artifacts are retained under
`selector-component-confirmation-invalid-missing-secure-eval/`.

The validation answers are reconstructed exactly from the persistent full labeled
public training set using the original stratified split and seed 42. Restoration must
match the already-persisted visible-train and hidden-validation IDs, order, text, and
manifest row counts before writing the root-only answer file. A scored uniform
submission and the isolation test must pass before confirmation restarts.

Because official private-test labels did not survive the restart and official-submit
performance is outside this experiment's estimand, an explicit runner-level
`solver.final_submit=false` gate is now fixed.
No placeholder labels or fabricated final score are used. Search-stage hidden
validation scoring, parent selection, budgets, root seeds, and endpoints are unchanged.

### Amendment 3 — benchmark-agnostic final-submit gate

After the first completed confirmation search (`20260816`), an audit found that the
requested `search.runner.submit_repeats=0` override was absent from the resolved
runner configuration: `submit_repeats` is sourced from the task configuration, and
the runner's separate `solver.final_submit` switch was honored only for NatureBench.
Consequently, one post-search self-validation evaluation ran after the fixed search
had already ended. It is not an official private-test score, is excluded from every
endpoint, and could not affect the already-recorded population states or decisions.

Before trajectories `20260817` and `20260818`, the existing final-submit switch was
made benchmark-agnostic and fixed to `solver.final_submit=false`. Focused tests must
show that MLE-Bench retains its historical default when the switch is absent, obeys
an explicit disable, and always obeys zero task-level repeats. This amendment changes
only post-search behavior; no search design, trajectory count, endpoint, or analysis
rule is changed. The `20260816` search remains eligible for the primary estimand, but
its post-search evaluation is explicitly excluded and will not be described as
official performance.

## Fixed run design

- Three sequential trajectories; no cross-run checkpoint reuse.
- Four generations, two attempted individuals per generation, one island.
- Maximum nine journal steps, 1,800 seconds of effective search time, 2,400
  seconds of model-plus-sandbox time, 600-second execution timeout.
- Experience memory and the original utility weights remain enabled:
  score `1.0`, positive delta `0.4`, novelty `0.25`.
- Crossover becomes eligible at generation 2 with the unmodified probability.
- One Debug attempt is allowed. Every failed/timeout attempt remains in the
  artifacts and summaries.
- Final official submission is disabled (`solver.final_submit=false`); it is not an outcome
  of this controller-activation audit.
- Common selection metric: the fixed hidden validation holdout already audited as
  inaccessible to generated programs. Official/private final scores are not used
  in parent selection.

## Legal support and statistical unit

The legal candidate set is the exact ordered island membership serialized at the
generation boundary immediately preceding the draw. It is not the journal/archive.
Each trace must match this serialized set exactly or the run is invalid for the
primary analysis.

Two individuals generated within one generation see the same pre-admission
population. Therefore the primary unit is a unique trajectory × generation × live
population state, not a child and not an individual draw. Duplicate traces from the
same state are retained for audit but counted once in the primary table.

## Primary estimands

For each unique legal state and component `c`:

`range(c) = max_i c_i - min_i c_i`

A component is discriminative only when its range is greater than `1e-12`.
Primary outputs are the exact number and fraction of unique states in which each
component discriminates, reported per trajectory and pooled descriptively.

The full utility is independently recomputed from the stored components and fixed
weights. Recorded probabilities are independently recomputed using the stored
temperature. Any mismatch larger than `1e-10` invalidates that state.

For states with an `Improve` draw, a secondary endpoint is the total-variation
distance between the recorded full-utility distribution and a score-only softmax
computed on the same legal candidates and temperature. Crossover states are
excluded from this one-parent probability interpretation because crossover samples
unordered two-parent actions without replacement.

## Failure, tie, and stopping rules

- Failed children are never silently removed from run-level counts.
- A run with no eligible multi-candidate state contributes zero eligible states and
  is reported, not replaced.
- Exact component ties are ties; no arbitrary winner is assigned.
- Runs stop only at the fixed solver limits or an integrity failure. Results are not
  inspected between seeds to decide whether to add/drop runs.
- If state support, metric provenance, prompt/seed provenance, or recomputation
  validation fails, generation freezes and no substantive claim is made from the
  affected run.

## Interpretation boundary

Permitted conclusion form:

> In N fresh short trajectories on this one task/model/budget, component X was
> non-discriminative at A/B exact legal population states; consequently it could
> not alter relative parent probabilities at those states.

Not permitted from this experiment:

- Frontis generally ignores or mishandles novelty/delta.
- A component causes poor downstream performance.
- A replacement selector would improve search.
- The published BF16 Frontis system behaves identically.
