#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 SEED" >&2
  exit 2
fi

seed="$1"
case "${seed}" in
  20260816|20260817|20260818) ;;
  *)
    echo "seed must be one of the preregistered confirmation seeds" >&2
    exit 2
    ;;
esac

research_root="${FRONTIS_RESEARCH_ROOT:-/workspace/frontis-research}"
evo_root="${FRONTIS_EVO_ROOT:-${research_root}/OpenRSI/OpenMLE-Evo}"
output_base="${FRONTIS_OUTPUT_ROOT:-${research_root}/outputs/selector-component-confirmation}"
output_root="${output_base}/seed-${seed}"
llm_models_url="${FRONTIS_LLM_MODELS_URL:-http://127.0.0.1:8080/v1/models}"
sandbox_health_url="${FRONTIS_SANDBOX_HEALTH_URL:-http://127.0.0.1:6580/health}"
secure_validation_root="${FRONTIS_SECURE_VALIDATION_ROOT:-/var/lib/frontis-pilot/secure/validation}"

if [[ -e "${output_root}" ]]; then
  echo "refusing to overwrite existing confirmation output: ${output_root}" >&2
  exit 1
fi

curl -fsS --max-time 5 "${llm_models_url}" >/dev/null
curl -fsS --max-time 5 "${sandbox_health_url}" >/dev/null
if [[ ! -s "${secure_validation_root}/spooky-author-identification/answers.csv" ]]; then
  echo "hidden validation answers are missing; restore and run the scored health check first" >&2
  exit 1
fi

export PYTHONHASHSEED="${seed}"
export OPENMLE_CONFIG_NAME=experiment/openmle_evo_smoke

cd "${evo_root}"
exec ./scripts/run_standard.sh \
  "output_dir=${output_root}" \
  "seed=${seed}" \
  "max_steps=9" \
  "time_budget=1800" \
  "model_plus_sandbox_time_budget=2400" \
  "+search.runner.solver.final_submit=false" \
  "search.runner.task_list=[spooky-author-identification]" \
  "search.runner.solver.num_generations=4" \
  "search.runner.solver.individuals_per_generation=2" \
  "search.runner.solver.max_debug_depth=1" \
  "search.runner.solver.num_generations_till_crossover=2"
