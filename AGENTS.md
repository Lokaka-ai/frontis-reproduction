# Frontis/OpenRSI Research Project Law

This file governs all research, analysis, implementation, and reporting in this workspace.

## Primary objective

Produce meaningful observations about Frontis/OpenRSI for research discussion, especially lab meetings. Scientific validity takes precedence over speed, novelty, and obtaining a positive result.

## Non-negotiable evidence standard

- Every substantive claim must be supported by an experiment or artifact whose estimand, intervention, candidate support, metric, and statistical unit match the claim.
- Before running an experiment, explicitly define the research question, estimand, legal action/candidate set, controls, outcome, failure treatment, unit of analysis, and interpretation boundary.
- Audit the actual implementation path rather than inferring behavior from names, comments, papers, or journal artifacts.
- Distinguish the journal/archive, live evolutionary population, legal parent/action support, operator choice, and stochastic policy.
- Use a common hidden evaluation metric for cross-candidate quality comparisons. Never substitute heterogeneous program-reported validation scores.
- Preserve baseline trajectories. Run interventions only in isolated, verifiably equivalent state copies.
- Separate exploratory/discovery evidence from confirmatory evidence. Do not reuse a discovered winner on the same states as confirmation.
- Treat correlated snapshots, repeated parents, shared trajectories, and multiple children from one state as dependent observations.
- Report failures, missing outcomes, compute, seeds, prompt provenance, and selection/admission events; do not silently exclude them.
- Persist enough state to reproduce the live population exactly, including island membership, admission/removal/migration events, solver counters, relevant caches, and RNG/model-sampling state.
- Verify that any checkpoint or clone reproduces the same legal candidates, utilities, prompts, and policy state before using it counterfactually.
- When model sampling is not explicitly seeded end to end, call repeats stochastic replicates rather than seed-paired trials.
- For multi-step search-value claims, force at most the predeclared intervention and then resume the complete normal population policy. A constrained descendant lineage is a different estimand.
- Prefer null or qualified conclusions over claims that exceed the evidence.

## Stop rule

Do not continue a scientific experiment when a design, implementation, metric, state-restoration, leakage, or logic problem could invalidate the intended claim. Freeze generation, investigate read-only, correct the design, and clearly supersede affected reports before resuming.

## Reporting rule

Lead with what the evidence directly establishes. Clearly separate:

1. verified descriptive observations;
2. supported but unconfirmed hypotheses;
3. claims the current sample cannot support.

Any superseded result must remain labeled as superseded wherever it could otherwise be reused or presented.
