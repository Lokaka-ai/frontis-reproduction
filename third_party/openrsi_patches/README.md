# Patch status

- `0001_population_state_observability.patch`: historical incremental patch.
- `0002_clone_equivalence_guards.patch`: historical incremental patch.
- `0003_llm_seed_and_prompt_provenance.patch`: **superseded; do not apply**. This
  intermediate export predates the completed call-specific seed schedule and its
  full test assertions.
- `0004_final_validity_instrumentation.patch`: authoritative cumulative patch against
  OpenRSI commit `ece6cbdf115ed72c3b62643a836504d77365e3a0`; apply it from the
  root of the `OpenRSI/` checkout. It is the only patch needed for
  reproduction. It contains the final source and
  test patch used for the valid confirmation trajectories. SHA-256:
  `897e18bc6f5c7cc1afecfe32ffeda9f803a7bad201edaf9d5de13cb27c072888`.

The cumulative patch includes exact live-population checkpoints, RNG and per-operator
LLM call-state restoration, prompt/completion/request provenance, deterministic
call-specific seeds, the benchmark-agnostic final-submit gate, and focused tests.
