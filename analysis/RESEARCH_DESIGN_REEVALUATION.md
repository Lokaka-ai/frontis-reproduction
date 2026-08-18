# Frontis pilot — complete research-design reevaluation

Status: **final audit; new experiments remain frozen pending redesign**

Audit date: 2026-08-16
Pinned OpenRSI commit: `ece6cbdf115ed72c3b62643a836504d77365e3a0`
Pinned MLE-Bench commit: `507f92e1138bb6e40dac5c6ee7a6758e6424bf97`

## Final verdict

The infrastructure and hidden evaluation are valid for a small local case study, but the main Checkpoint 4/search-value interpretation is not valid as an audit of Frontis's parent selector.

The decisive error is candidate support:

> The probe treated every successful journal node as a selectable parent. Frontis actually samples from the live island population, which contains only admitted nodes. The recurring probe oracle, `f28d7d…`, was never admitted and was not a legal parent after it was generated.

Consequences:

- Retract the claim that Frontis repeatedly chose a worse parent than `f28d7d…`.
- Retain the `f28d7d…` result only as an archive-wide, forced-`Improve` repairability observation.
- Retract selector-component ablation conclusions computed over the archive-wide probe support.
- Retain the novelty audit only as a descriptive audit of stored node-level family-rarity metadata versus one lexical code-distance measure.
- Do not start the proposed memory ablation, replication, or rollout in their previous forms.

The strongest corrected scientific observation is:

> In this baseline, every legal live-population parent was in the same `sklearn` family and had zero positive delta. Novelty and delta were therefore constant across candidates at each recorded selection event. The implemented utility controller reduced to score-only ranking plus an equal constant. This trajectory cannot identify whether novelty or delta helps or hurts selection.

## What passed verification

### Common hidden score provenance — pass

The programs printed heterogeneous self-validation scores, but those values were not used as `node.metric.value` in this run.

Exact path:

1. The sandbox returned `sandbox_valid_score` from the fixed hidden holdout.
2. `SandboxMLEBenchTask._annotate_eval_scores` set `selection_score` to that hidden value.
3. `step_task` emitted it as `VALIDATION_FITNESS`.
4. `Evolutionary.parse_eval_result` assigned it to `node.metric`.
5. The probe recorded `child.metric.value` as `child_score`.

Checkpoint evidence includes:

- baseline node `8f743f…`: hidden metric `0.36072`, self-validation `0.4075925`;
- one probe child: hidden `child_score = 0.39695`, self-validation `0.4249740`.

Thus absolute child scores are comparable on the same hidden holdout. The utility trace label `score_source: self_validation` is misleading in this configuration; the metric payload itself records `selection_score_source: sandbox_valid_score`.

### Hidden-label isolation — pass with scope limits

- Generated code did not receive hidden labels.
- Hidden labels were held outside the generated-code mount and used by the sandbox scorer.
- The fixed holdout was reconstructed from labeled Kaggle training data; this is acceptable for a local pilot if it remains inaccessible to generated programs.
- This is a custom local evaluation, one split, one task, and a quantized model—not canonical benchmark reproduction.

### Probe-tree isolation — pass

- The 24 probe children were stored separately.
- They were not appended to the baseline journal or live population.
- The baseline trajectory was not contaminated by counterfactual outcomes.

### Prefix journal isolation — no future population evidence found

The probe rebuilt a prefix `Journal`, truncated child references, and attached only experience cards with `step_id <= snapshot_step`. Operator memory and strategy-board statistics are recomputed from the current prefix journal and its attached cards.

The only external cache is per-node rich-summary memory. Its source prompt uses the node and its parent, not later descendants or a later board. No future node identifier appeared in the stored snapshot-4 child artifacts.

Qualification: exact rendered probe prompts were not persisted as first-class artifacts, and cached summaries may have been generated later than the reconstructed snapshot. This is not evidence of future outcome leakage, but future experiments must store exact prompts and either regenerate summaries inside each state clone or prove cache equivalence.

### LLM seed propagation — fail

The probe called Python and NumPy seeding, which controls local operations such as package-order shuffling. It did not include a `seed` in the `GenericLLM` generation kwargs, and the LiteLLM backend forwarded no model seed to llama.cpp.

Therefore:

- the recorded probe `seed` is not an LLM sampling seed;
- repeats are stochastic replicates, not seed-paired generations;
- future work must pass and log an explicit request seed if llama.cpp supports it, or declare model sampling uncontrolled and use independent replicate inference.

## The critical population-support error

Frontis maintains both:

- a journal/archive containing all evaluated nodes; and
- a live `SolutionsDatabase` of island members eligible for parent sampling.

After initialization, a child is admitted only when it is at least as good as its island's current average. With one island in this run, several valid journal nodes were rejected. Checkpoint 4 instead used all nonbuggy journal nodes:

```python
valid_nodes = [node for node in solver.journal.nodes if ...]
```

It never reconstructed the live `SolutionsDatabase` before choosing probe parents or computing utility.

### Correct live support versus probed support

| Snapshot | Legal live parents | Legal parents probed | Archive-only parents probed | Corrected inference |
|---|---|---|---|---|
| 4 | `8f743f…`, `c711d9…` | both | `be37d6…`, `f28d7d…` | Full legal support, but only two offspring each |
| 5 | `8f743f…`, `c711d9…`, `b7131e…` | only `8f743f…` | `be37d6…`, `f6de56…`, `f28d7d…` | No legal-parent comparison |
| 9 | `8f743f…`, `c711d9…`, `b7131e…`, `3d6198…` | `b7131e…`, `3d6198…` | `f6de56…`, `f28d7d…` | Partial legal support only |

At snapshot 4, where legal support was complete, the higher-utility `8f743f…` parent had mean hidden child loss `0.464575`, versus `0.771730` for lower-utility `c711d9…`. This one tiny block aligns with utility rather than showing the reported anomaly.

At snapshot 9, the two probed legal parents favored lower-utility `b7131e…` (`0.406685`) over higher-utility `3d6198…` (`0.428930`), but the other two legal parents were not probed. No selector oracle or regret can be identified.

### Checkpoint restoration is itself not population-faithful

The saved checkpoint contains journal and scalar solver state, not island membership. `_restore_solution_database_from_journal` replays all good journal nodes through the average-fitness admission filter from an empty island. That is not equivalent to the uninterrupted run, which seeded the first generation without that filter and then admitted later generations in batches.

This means a future population-resume experiment cannot rely on `load_checkpoint()` alone. It must either:

- persist exact island membership and relevant population statistics at decision boundaries; or
- reconstruct them from a verified event log that preserves initial seeding, generation batching, admissions, removals, migrations, and RNG state.

## Checkpoint-by-checkpoint reevaluation

### Checkpoints 1–2: setup and smoke test

Valid for infrastructure only: model serving, sandbox execution, scoring, artifact creation, and filesystem isolation worked. They provide no evidence about search quality.

### Checkpoint 3: one baseline trajectory

Valid descriptive facts:

- ten generated nodes plus a synthetic root;
- eight successful and two failed evaluations;
- the first Draft established the best hidden validation loss and was not beaten;
- actual operator choices, parent-selection traces, runtimes, failures, and lineage are preserved.

Limits:

- one task, one trajectory, one quantization, and one search history;
- no expected-performance or task-general conclusion;
- model sampling was not explicitly seeded;
- final/static family failure rate is not a prospective covariate;
- checkpoint resume does not faithfully preserve the live population.

### Checkpoint 4: counterfactual parent probe

What remains valid:

- a 24-child forced-`Improve` archive-wide probe;
- common hidden-score evaluation;
- two stochastic offspring per selected archive node;
- an observation that `f28d7d…` was highly repairable under the stored cross-branch memory context.

What is invalid:

- calling `f28d7d…` a counterfactual Frontis parent choice;
- 3/3 selector top-1 disagreement;
- selector regret computed against archive-only parents;
- probability-weighted policy value after renormalizing over the four probed archive nodes.

Additional limitations remain:

- nested correlated snapshots from one trajectory;
- repeated focal parent, not three independent discoveries;
- forced `Improve` only;
- top-1 ranking is not the stochastic softmax action;
- two offspring per parent;
- no explicit LLM seed;
- original handoff improvement formula had the wrong sign for minimized loss.

### Checkpoint 5: novelty/family audit

The lexical analysis is reproducible but narrow:

- ten nodes, nine `sklearn` and one `ensemble`;
- pairwise comparisons share nodes and are not independent;
- token sets ignore order, frequency, and literal values;
- high/low novelty was a post-hoc median split;
- family failure rate was final/static.

More importantly, the audited `novelty_score` is node-creation metadata based on prior experiment cards. Parent selection recomputes a novelty component over the live island candidates. In the actual legal pools, all candidates were `sklearn`, so all received the same novelty component at each event (`0.7071` with two candidates and `0.57735` with three). Novelty did not discriminate among legal parents in this baseline.

Valid conclusion:

> Stored family-rarity metadata was weakly aligned with one lexical distance measure in this tiny archive. The trajectory provides no evidence about whether the selector's novelty term improved or harmed parent decisions.

### Search-value audit and component ablations

The conceptual correction from parent-relative improvement to absolute child quality was sound. Failure sensitivity and hidden incumbent comparisons were also appropriate.

However, the numerical selector conclusions are invalid because:

- outcomes were measured mostly on archive-only or incompletely covered legal support;
- utilities were recomputed over all valid journal nodes rather than the live island candidates;
- softmax probabilities were renormalized over four probed archive nodes;
- component ablations were exploratory and used incomparable utility scales at a fixed temperature;
- delta and novelty were actually constant across legal candidates in this trajectory.

The code-feature convergence result remains descriptive, not causal evidence that memory caused repairability.

## Correct design for the next experiments

### Phase 0 — repair observability before new scientific runs

1. Persist exact island membership at every parent-decision boundary.
2. Persist population admission/removal/migration events and RNG states.
3. Persist exact rendered prompts, rich-summary provenance, generation kwargs, and request seeds.
4. Define decision states at authentic operator/parent-choice boundaries, not arbitrary journal lengths.
5. Verify clone equivalence by asking two untouched clones for the same legal candidate set and utility vector before any intervention.

### Phase 1 — same-task legal-support replication

Use independent trajectories and predeclared decision states.

- Probe every legal parent when the pool is small.
- For crossover, treat unordered legal parent pairs as actions; do not reduce a two-parent action to a single-parent comparison.
- Use the operator that the policy chose, or stratify estimands explicitly by operator.
- Use at least three model generations per legal action; more are preferable for failure-prone actions.
- Primary outcome: common hidden score, with a prospective failure rule.
- Statistical unit: independent trajectory/decision-state block, not child.
- Estimate the stochastic policy's expected one-step value over its true legal support.
- Keep discovery trajectories separate from confirmation trajectories.

This phase can test utility alignment. It should not yet test a new learned selector.

### Phase 2 — mechanism test for cross-branch memory

Only if Phase 1 identifies a repeatable legal-parent anomaly:

- full authentic memory;
- parent-local memory with cross-branch evidence removed;
- structure/token-matched placebo memory with no outcome-bearing cross-branch content.

Use fresh held-out states, predeclare tie handling, save prompts, and measure hidden child score, failures, runtime, and predeclared transferred code features. Interpret this as a memory-content effect, not a universal repairability mechanism.

### Phase 3 — one-action intervention with global policy resumption

To test actual search value:

1. Clone an exact live population state into matched arms.
2. Force only the first legal action: parent for `Improve`, or pair for `Crossover`.
3. Evaluate and admit/reject the resulting child exactly as normal Frontis would.
4. Resume the complete unmodified global policy with access to all surviving candidates.
5. Run for a fixed number of evaluations or matched compute budget.
6. Measure final/best population score, incumbent-update probability, area under best-score curve, failures, parent choices, and compute.

A forced lineage `A → A1 → A2` is only a secondary branch-potential diagnostic because Frontis may return to any surviving population member on the next step.

### Phase 4 — external validation

After a legal-support anomaly and mechanism replicate within task, freeze an intervention rule and test it on at least one additional inexpensive task. A new task now would add heterogeneity before the current estimand is valid.

## Final go/no-go checklist

Verified:

- [x] hidden score provenance and cross-program comparability;
- [x] probe children isolated from baseline;
- [x] no direct evidence of future journal-node leakage;
- [x] model seed not propagated and therefore declared uncontrolled;
- [x] legal population-support error identified;
- [x] built-in checkpoint-resume mismatch identified.

Required before new scientific generation:

- [ ] exact population-state persistence or verified replay;
- [ ] authentic decision-boundary extraction;
- [ ] full legal action support or a valid sampling/off-policy design;
- [ ] exact prompt and rich-summary provenance logging;
- [ ] explicit LLM request seed, or an independent-replicate design;
- [ ] predeclared outcome, failure rule, tie rule, operator stratum, and uncertainty method;
- [ ] discovery/confirmation separation;
- [ ] verified normal global-policy resumption after the forced first action.

## Decision

**No-go on the previously proposed experiments.**

The next implementation should be a population-state observability and clone-equivalence repair, followed by a fresh legal-support same-task replication. Existing Checkpoint 4 generations should not be discarded, but they must be labeled archive-wide mechanism probes rather than evidence of Frontis selector failure.
