#!/usr/bin/env python3
"""Research-only counterfactual Improve probes from saved OpenMLE-Evo snapshots.

Probe children are never appended to the loaded journal or solution database.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import yaml


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle) or {})


def build_dataclass(cls: type[Any], payload: dict[str, Any]) -> Any:
    names = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in names})


def build_operator(payload: dict[str, Any], classes: dict[str, type[Any]]) -> Any:
    client = build_dataclass(classes["ClientConfig"], payload["llm"]["client"])
    llm = classes["GenericLLMConfig"](
        client=client,
        generation_kwargs=dict(payload["llm"].get("generation_kwargs") or {}),
    )

    def prompt(name: str) -> Any:
        return build_dataclass(classes["JinjaPromptConfig"], payload.get(name) or {})

    return classes["OperatorConfig"](
        llm=llm,
        system_message_prompt_template=prompt("system_message_prompt_template"),
        init_user_message_prompt_template=prompt("init_user_message_prompt_template"),
        user_message_prompt_template=prompt("user_message_prompt_template"),
    )


def usage(metrics: list[dict[str, Any]]) -> dict[str, float]:
    result = {
        "prompt_tokens": 0.0,
        "completion_tokens": 0.0,
        "total_tokens": 0.0,
        "model_runtime": 0.0,
    }
    for metric in metrics:
        item = dict(metric.get("usage") or {})
        result["prompt_tokens"] += float(item.get("prompt_tokens") or 0)
        result["completion_tokens"] += float(item.get("completion_tokens") or 0)
        result["total_tokens"] += float(item.get("total_tokens") or 0)
        result["model_runtime"] += float(item.get("latency") or 0)
    return result


def utility_map(
    nodes: list[Any], cards: list[dict[str, Any]], experience: dict[str, Any], fn: Any
) -> dict[str, dict[str, Any]]:
    parent_cfg = dict(experience.get("parent_selection") or {})
    items = fn(
        nodes,
        lower_is_better=True,
        previous_cards=cards,
        weights=dict(parent_cfg.get("weights") or {}),
        component_normalization=dict(parent_cfg.get("component_normalization") or {}),
        temperature=1.0,
    )
    return {str(item["node_id"]): item for item in items}


def choose_score_coverage(nodes: list[Any], count: int = 4) -> list[Any]:
    ordered = sorted(nodes, key=lambda node: (float(node.metric.value), str(node.id)))
    if len(ordered) <= count:
        return ordered
    positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return [ordered[position] for position in positions]


def write_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-task-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--snapshot-steps", default="4,5,9")
    parser.add_argument("--parents-per-snapshot", type=int, default=4)
    parser.add_argument("--offspring-per-parent", type=int, default=2)
    parser.add_argument(
        "--dojo-root",
        type=Path,
        default=Path(
            os.environ.get(
                "FRONTIS_DOJO_ROOT",
                "/workspace/frontis-research/OpenRSI/OpenMLE-Evo/third_party/aira-evo",
            )
        ),
        help="Path to the aira-evo checkout containing src/ and examples/mle_bench/.",
    )
    args = parser.parse_args()

    baseline = args.baseline_task_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LOGGING_DIR", str(output / "logs"))
    snapshots = [int(item) for item in args.snapshot_steps.split(",") if item.strip()]
    dojo_root = args.dojo_root.resolve()
    sys.path[:0] = [str(dojo_root / "src"), str(dojo_root / "examples/mle_bench")]

    from base_task import SandboxMLEBenchTask
    from dojo.config_dataclasses.client.base import ClientConfig
    from dojo.config_dataclasses.interpreter.python import PythonInterpreterConfig
    from dojo.config_dataclasses.llm.generic_llm import GenericLLMConfig
    from dojo.config_dataclasses.llm.jinjaprompt import JinjaPromptConfig
    from dojo.config_dataclasses.logger import LoggerConfig
    from dojo.config_dataclasses.operators.base import OperatorConfig
    from dojo.config_dataclasses.operators.memory import MemoryOpConfig
    from dojo.config_dataclasses.solver.evo import EvolutionarySolverConfig
    from dojo.core.interpreters.python import PythonInterpreter
    from dojo.core.solvers.utils.journal import Journal
    from dojo.solvers.evo.evo import Evolutionary
    from dojo.solvers.evo.experience import compute_parent_utilities
    from dojo.utils.logger import config_logger

    classes = {
        "ClientConfig": ClientConfig,
        "GenericLLMConfig": GenericLLMConfig,
        "JinjaPromptConfig": JinjaPromptConfig,
        "OperatorConfig": OperatorConfig,
    }
    saved = load_json(baseline / "aira_evo/dojo_config.json")
    solver_payload = copy.deepcopy(saved["solver"])
    solver_payload["operators"] = {
        name: build_operator(payload, classes)
        for name, payload in solver_payload["operators"].items()
    }
    solver_payload["memory"] = build_dataclass(MemoryOpConfig, solver_payload["memory"])
    solver_payload["debug_memory"] = build_dataclass(
        MemoryOpConfig, solver_payload["debug_memory"]
    )
    solver_payload["checkpoint_path"] = str(baseline / "aira_evo/checkpoint")
    solver_payload["time_limit_secs"] = 86400
    solver_payload["step_limit"] = 1000
    solver_cfg = build_dataclass(EvolutionarySolverConfig, solver_payload)

    logger_cfg = LoggerConfig(
        output_dir=str(output / "logs"),
        use_console=True,
        use_wandb=False,
        use_json=True,
        print_config=False,
        write_env_vars=False,
    )
    config_logger(SimpleNamespace(logger=logger_cfg))

    task_cfg = load_yaml(
        baseline.parents[1] / ".hydra/airaevo_tasks/spooky-author-identification/config.yaml"
    )
    task_cfg["sandbox"]["verify_tls"] = True
    if not os.environ.get("SANDBOX_CPU_API_KEY"):
        raise SystemExit("SANDBOX_CPU_API_KEY must be set for the sandbox endpoint")
    task = SandboxMLEBenchTask(task_cfg, time_budget=None)
    interpreter_cfg = build_dataclass(PythonInterpreterConfig, saved["interpreter"])
    interpreter_cfg.working_dir = str(output / "interpreter_workspace")
    interpreter = PythonInterpreter(
        interpreter_cfg, data_dir=Path(str(task_cfg["data_dir"]))
    )
    state, task_info = task.prepare(solver_interpreter=interpreter, eval_interpreter=None)
    solver = Evolutionary(solver_cfg, task_info=task_info)
    solver.load_checkpoint()
    full_export = solver.journal.node_list()
    baseline_cards = [
        json.loads(line)
        for line in (baseline / "experience_cards.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    card_by_id = {str(card["node_id"]): card for card in baseline_cards}
    for node in solver.journal.nodes:
        if str(node.id) in card_by_id:
            node.experience_card = card_by_id[str(node.id)]

    manifest = {
        "schema_version": 1,
        "baseline_task_dir": str(baseline),
        "baseline_checkpoint": str(baseline / "aira_evo/checkpoint"),
        "snapshot_steps": snapshots,
        "parents_per_snapshot": args.parents_per_snapshot,
        "offspring_per_parent": args.offspring_per_parent,
        "operator": "improve",
        "model_generation_kwargs": saved["solver"]["operators"]["improve"]["llm"][
            "generation_kwargs"
        ],
        "parent_selection_weights": saved["solver"]["experience"]["parent_selection"],
        "parent_choice_rule": "score-ranked coverage: equally spaced ranks including best and worst",
        "probe_children_committed": False,
        "metric_direction": "lower_is_better",
        "handoff_formula": "max(0, child_score - parent_score)",
        "direction_corrected_formula": "max(0, parent_score - child_score)",
    }
    (output / "probe_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    results_path = output / "probe_results.jsonl"
    completed: set[str] = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(str(json.loads(line)["probe_id"]))

    for snapshot_step in snapshots:
        prefix_size = snapshot_step + 2
        prefix_export = copy.deepcopy(full_export[:prefix_size])
        for exported_node in prefix_export:
            exported_node["children"] = [
                child_step
                for child_step in exported_node.get("children", [])
                if int(child_step) < prefix_size
            ]
        solver.journal = Journal.from_export_data({"nodes": prefix_export})
        prefix_cards = [card for card in baseline_cards if int(card["step_id"]) <= snapshot_step]
        for node in solver.journal.nodes:
            if str(node.id) in card_by_id:
                node.experience_card = card_by_id[str(node.id)]
        valid_nodes = [
            node
            for node in solver.journal.nodes
            if not solver.journal.is_root_node(node)
            and not bool(node.is_buggy)
            and node.metric is not None
            and node.metric.value is not None
        ]
        utilities = utility_map(
            valid_nodes, prefix_cards, saved["solver"]["experience"], compute_parent_utilities
        )
        selected = choose_score_coverage(valid_nodes, args.parents_per_snapshot)
        snapshot_best = min(float(node.metric.value) for node in valid_nodes)
        snapshot_payload = {
            "snapshot_step": snapshot_step,
            "candidate_count": len(valid_nodes),
            "snapshot_best_score": snapshot_best,
            "candidate_utilities": list(utilities.values()),
            "selected_parent_ids": [node.id for node in selected],
        }
        (output / f"snapshot_{snapshot_step}.json").write_text(
            json.dumps(snapshot_payload, indent=2), encoding="utf-8"
        )

        for parent in selected:
            parent_score = float(parent.metric.value)
            utility = utilities[str(parent.id)]
            for repeat in range(args.offspring_per_parent):
                probe_id = f"s{snapshot_step}-{parent.id}-r{repeat}"
                if probe_id in completed:
                    continue
                probe_dir = output / "children" / probe_id
                probe_dir.mkdir(parents=True, exist_ok=True)
                seed = 420000 + snapshot_step * 1000 + int(parent.step) * 10 + repeat
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                solver.state.current_step = snapshot_step + 2
                solver.state.running_time = 0.0
                started = time.monotonic()
                child = None
                error = None
                try:
                    child = solver._improve(parent)
                    generation_finished = time.monotonic()
                    state, eval_result = task.step_task(dict(state), child.code)
                    solver.parse_eval_result(child, eval_result)
                    finished = time.monotonic()
                    aux = dict(child.metric.info or {}) if child.metric is not None else {}
                    child_score = (
                        float(child.metric.value)
                        if child.metric is not None and child.metric.value is not None
                        else None
                    )
                    status = str(aux.get("status") or ("failed" if child.is_buggy else "unknown"))
                    sandbox_runtime = float(aux.get("run_time") or child.exec_time or 0.0)
                    raw_log = str(aux.get("raw_run_log") or "")
                    clear_log = str(aux.get("clear_run_log") or child.term_out or "")
                    feedback = str(aux.get("feedback") or child.analysis or "")
                except Exception as exc:
                    generation_finished = time.monotonic()
                    finished = generation_finished
                    child_score = None
                    status = "probe_exception"
                    sandbox_runtime = 0.0
                    raw_log = ""
                    clear_log = ""
                    feedback = ""
                    error = f"{type(exc).__name__}: {exc}"

                operator_usage = usage(child.operators_metrics if child is not None else [])
                model_runtime = operator_usage["model_runtime"] or (
                    generation_finished - started
                )
                row = {
                    "probe_id": probe_id,
                    "snapshot_step": snapshot_step,
                    "snapshot_best_score": snapshot_best,
                    "parent_id": str(parent.id),
                    "parent_step": int(parent.step) - 1,
                    "parent_score": parent_score,
                    "parent_utility": utility["utility"],
                    "parent_probability": utility["probability"],
                    "parent_score_component": utility["score_component"],
                    "parent_delta_component": utility["delta_component"],
                    "parent_novelty_component": utility["novelty_component"],
                    "repeat": repeat,
                    "seed": seed,
                    "operator": "improve",
                    "child_id": str(child.id) if child is not None else None,
                    "child_score": child_score,
                    "status": status,
                    "is_buggy": bool(child.is_buggy) if child is not None else True,
                    "direction_corrected_improvement": (
                        max(0.0, parent_score - child_score)
                        if child_score is not None
                        else 0.0
                    ),
                    "handoff_formula_positive_improvement": (
                        max(0.0, child_score - parent_score)
                        if child_score is not None
                        else 0.0
                    ),
                    "child_improves_parent": (
                        child_score is not None and child_score < parent_score
                    ),
                    "child_new_global_best": (
                        child_score is not None and child_score < snapshot_best
                    ),
                    "model_runtime": model_runtime,
                    "sandbox_runtime": sandbox_runtime,
                    "execution_runtime": sandbox_runtime,
                    "wall_runtime": finished - started,
                    "prompt_tokens": int(operator_usage["prompt_tokens"]),
                    "completion_tokens": int(operator_usage["completion_tokens"]),
                    "total_tokens": int(operator_usage["total_tokens"]),
                    "error": error,
                    "code_path": str(probe_dir / "child_code.py"),
                    "response_path": str(probe_dir / "response.md"),
                    "reasoning_path": str(probe_dir / "reasoning_content.md"),
                    "feedback_path": str(probe_dir / "feedback.txt"),
                    "raw_log_path": str(probe_dir / "raw_run_log.txt"),
                    "clear_log_path": str(probe_dir / "clear_run_log.txt"),
                }
                metric = child.operators_metrics[0] if child and child.operators_metrics else {}
                completion_text = str(metric.get("completion_text") or "")
                reasoning = str((metric.get("usage") or {}).get("reasoning_content") or "")
                (probe_dir / "child_code.py").write_text(
                    child.code if child is not None else "", encoding="utf-8"
                )
                (probe_dir / "response.md").write_text(completion_text, encoding="utf-8")
                (probe_dir / "reasoning_content.md").write_text(reasoning, encoding="utf-8")
                (probe_dir / "feedback.txt").write_text(feedback, encoding="utf-8")
                (probe_dir / "raw_run_log.txt").write_text(raw_log, encoding="utf-8")
                (probe_dir / "clear_run_log.txt").write_text(clear_log, encoding="utf-8")
                (probe_dir / "result.json").write_text(
                    json.dumps(row, indent=2), encoding="utf-8"
                )
                write_row(results_path, row)
                print(json.dumps({key: row[key] for key in (
                    "probe_id", "parent_score", "parent_utility", "child_score",
                    "status", "child_improves_parent", "sandbox_runtime"
                )}), flush=True)

    rows = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    columns = list(rows[0]) if rows else []
    with (output / "probe_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"completed_probes": len(rows), "output_dir": str(output)}))


if __name__ == "__main__":
    main()
