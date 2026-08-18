#!/usr/bin/env python3
"""Audit selector activation only on exact serialized legal populations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


TOL = 1e-12
RECOMPUTE_TOL = 1e-10
COMPONENTS = ("score_component", "delta_component", "novelty_component")


def softmax(values: list[float], temperature: float) -> list[float]:
    temperature = max(float(temperature or 1.0), 1e-8)
    shifted = [(value - max(values)) / temperature for value in values]
    exps = [math.exp(value) for value in shifted]
    total = sum(exps)
    return [value / total for value in exps]


def argmax_ids(candidates: list[dict[str, Any]], values: list[float]) -> list[str]:
    maximum = max(values)
    return [
        str(candidate["node_id"])
        for candidate, value in zip(candidates, values)
        if abs(value - maximum) <= RECOMPUTE_TOL
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_one(root: Path, name: str) -> Path:
    matches = sorted(root.rglob(name))
    if len(matches) != 1:
        raise ValueError(f"expected one {name} under {root}, found {len(matches)}")
    return matches[0]


def find_primary_journal(root: Path) -> Path:
    matches = [
        path
        for path in sorted(root.rglob("checkpoint/journal.jsonl"))
        if "decision_states" not in path.parts
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one primary journal under {root}, found {len(matches)}")
    return matches[0]


def canonical_candidates(trace: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = list(trace.get("candidates") or [])
    if len(candidates) < 2:
        raise ValueError("primary states require at least two legal candidates")
    node_ids = [str(item["node_id"]) for item in candidates]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("candidate trace contains duplicate node IDs")
    return candidates


def trace_signature(trace: dict[str, Any]) -> str:
    payload = {
        "island_id": int(trace["island_id"]),
        "temperature": float(trace["temperature"]),
        "candidates": [
            {
                "node_id": str(item["node_id"]),
                **{name: float(item[name]) for name in COMPONENTS},
                "utility": float(item["utility"]),
                "probability": float(item["probability"]),
            }
            for item in canonical_candidates(trace)
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_prompt_seed(run_root: Path, expected_seed: int) -> dict[str, int]:
    journal_path = find_primary_journal(run_root)
    nodes = load_jsonl(journal_path)
    traced_calls = 0
    operator_seeds: dict[str, list[int]] = {}
    for node in nodes:
        operators = list(node.get("operators_used") or [])
        metrics = list(node.get("operators_metrics") or [])
        if len(operators) != len(metrics):
            raise ValueError(f"operator/metric trace length mismatch for node {node.get('id')}")
        for operator, metric in zip(operators, metrics):
            if not isinstance(metric, dict) or "usage" not in metric:
                continue
            traced_calls += 1
            for key in ("prompt_messages", "completion_text", "generation_kwargs"):
                if key not in metric:
                    raise ValueError(f"missing {key} in LLM trace for node {node.get('id')}")
            request_seed = int(metric["generation_kwargs"].get("seed", -1))
            if request_seed < expected_seed:
                raise ValueError(f"LLM request seed below root seed for node {node.get('id')}")
            operator_seeds.setdefault(str(operator), []).append(request_seed)
    if traced_calls == 0:
        raise ValueError("no persisted LLM request traces found")
    for operator, seeds in operator_seeds.items():
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate call-specific request seed for operator {operator}")
    return {
        "journal_nodes": len(nodes),
        "traced_llm_calls": traced_calls,
        "traced_llm_operators": len(operator_seeds),
    }


def preceding_population(run_root: Path, generation_id: int, island_id: int) -> list[str]:
    pattern = f"generation_{generation_id:04d}_step_*/population_state.json"
    matches = sorted(run_root.rglob(pattern))
    if not matches:
        raise ValueError(
            f"expected a preceding population for generation {generation_id}, found none"
        )
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in matches]
    if any(
        int(payload["solver_state"]["current_generation"]) != generation_id
        for payload in payloads
    ):
        raise ValueError("serialized generation does not match decision generation")
    canonical_populations = {
        json.dumps(payload["population"], sort_keys=True, separators=(",", ":"))
        for payload in payloads
    }
    if len(canonical_populations) != 1:
        raise ValueError(
            f"generation {generation_id} has multiple non-equivalent population snapshots"
        )
    islands = list(payloads[0]["population"]["islands"])
    island = next(item for item in islands if int(item["island_id"]) == island_id)
    return [str(node_id) for node_id in island["node_ids"]]


def analyze_run(run_root: Path, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_counts = validate_prompt_seed(run_root, seed)
    cards_path = find_one(run_root, "experience_cards.jsonl")
    cards = load_jsonl(cards_path)
    raw_traces: list[tuple[dict[str, Any], dict[str, Any]]] = []
    failed_cards = 0
    for card in cards:
        if bool(card.get("is_buggy")) or str(card.get("status", "")).lower() not in {
            "ok", "passed", "success", "successful", "valid"
        }:
            failed_cards += 1
        trace = card.get("parent_selection_trace")
        if not isinstance(trace, dict) or not trace.get("enabled"):
            continue
        if len(trace.get("candidates") or []) < 2:
            continue
        raw_traces.append((card, trace))

    grouped: dict[tuple[int, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for card, trace in raw_traces:
        generation_id = int(card["generation_id"])
        grouped.setdefault((generation_id, trace_signature(trace)), []).append((card, trace))

    rows: list[dict[str, Any]] = []
    for (generation_id, signature), draws in sorted(grouped.items()):
        trace = draws[0][1]
        candidates = canonical_candidates(trace)
        candidate_ids = [str(item["node_id"]) for item in candidates]
        island_id = int(trace["island_id"])
        live_ids = preceding_population(run_root, generation_id, island_id)
        if candidate_ids != live_ids:
            raise ValueError(
                f"legal-support mismatch at seed {seed}, generation {generation_id}: "
                f"trace={candidate_ids}, population={live_ids}"
            )

        weights = dict(trace.get("weights") or {})
        full_utilities = []
        for item in candidates:
            expected_utility = (
                float(weights["score"]) * float(item["score_component"])
                + float(weights["delta"]) * float(item["delta_component"])
                + float(weights["novelty"]) * float(item["novelty_component"])
                - float(item.get("official_score_missing_penalty", 0.0))
            )
            if abs(expected_utility - float(item["utility"])) > RECOMPUTE_TOL:
                raise ValueError("stored utility does not match stored components")
            full_utilities.append(expected_utility)

        temperature = float(trace["temperature"])
        full_probabilities = softmax(full_utilities, temperature)
        recorded_probabilities = [float(item["probability"]) for item in candidates]
        if max(abs(a - b) for a, b in zip(full_probabilities, recorded_probabilities)) > RECOMPUTE_TOL:
            raise ValueError("stored probability does not match recomputed softmax")

        score_utilities = [float(weights["score"]) * float(item["score_component"]) for item in candidates]
        score_probabilities = softmax(score_utilities, temperature)
        tv_score_only = 0.5 * sum(
            abs(a - b) for a, b in zip(full_probabilities, score_probabilities)
        )
        operators = sorted({str(card.get("operator")) for card, _ in draws})
        method_families = [str(item.get("method_family_auto")) for item in candidates]
        full_argmax = argmax_ids(candidates, full_utilities)
        score_argmax = argmax_ids(candidates, score_utilities)
        row: dict[str, Any] = {
            "seed": seed,
            "generation_id": generation_id,
            "state_signature": signature,
            "island_id": island_id,
            "candidate_count": len(candidates),
            "candidate_ids": "|".join(candidate_ids),
            "draw_count": len(draws),
            "operators": "|".join(operators),
            "has_improve_draw": int("improve" in operators),
            "method_families": "|".join(method_families),
            "distinct_method_families": len(set(method_families)),
            "temperature": temperature,
            "tv_full_vs_score_only": tv_score_only if "improve" in operators else None,
            "full_argmax_ids": "|".join(full_argmax) if "improve" in operators else None,
            "score_argmax_ids": "|".join(score_argmax) if "improve" in operators else None,
            "argmax_set_changed": int(full_argmax != score_argmax) if "improve" in operators else None,
        }
        for component in COMPONENTS:
            values = [float(item[component]) for item in candidates]
            value_range = max(values) - min(values)
            row[f"{component}_range"] = value_range
            row[f"{component}_discriminative"] = int(value_range > TOL)
        rows.append(row)

    summary = {
        "seed": seed,
        "cards": len(cards),
        "failed_cards": failed_cards,
        "raw_multi_candidate_draws": len(raw_traces),
        "unique_legal_states": len(rows),
        **prompt_counts,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for item in args.run:
        seed_text, path_text = item.split("=", 1)
        rows, summary = analyze_run(Path(path_text).resolve(), int(seed_text))
        all_rows.extend(rows)
        runs.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    fieldnames = list(all_rows[0]) if all_rows else ["seed", "generation_id"]
    with (args.output_dir / "legal_state_components.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)

    aggregate: dict[str, Any] = {
        "schema_version": 1,
        "tolerance": TOL,
        "runs": runs,
        "unique_legal_states": len(all_rows),
        "trajectories_with_eligible_states": len({row["seed"] for row in all_rows}),
    }
    for component in COMPONENTS:
        count = sum(int(row[f"{component}_discriminative"]) for row in all_rows)
        aggregate[f"{component}_discriminative_states"] = count
        aggregate[f"{component}_discriminative_fraction"] = count / len(all_rows) if all_rows else None
    improve_tvs = [
        float(row["tv_full_vs_score_only"])
        for row in all_rows
        if row["tv_full_vs_score_only"] is not None
    ]
    aggregate["improve_states"] = len(improve_tvs)
    aggregate["improve_states_full_equals_score_only"] = sum(value <= RECOMPUTE_TOL for value in improve_tvs)
    aggregate["improve_states_full_differs_from_score_only"] = sum(value > RECOMPUTE_TOL for value in improve_tvs)
    aggregate["max_tv_full_vs_score_only_on_improve"] = max(improve_tvs) if improve_tvs else None
    aggregate["mean_tv_full_vs_score_only_on_improve"] = (
        sum(improve_tvs) / len(improve_tvs) if improve_tvs else None
    )
    aggregate["improve_states_argmax_set_changed"] = sum(
        int(row["argmax_set_changed"])
        for row in all_rows
        if row["argmax_set_changed"] is not None
    )
    aggregate["single_family_states"] = sum(
        int(row["distinct_method_families"] == 1) for row in all_rows
    )
    aggregate["observed_method_families"] = sorted(
        {
            family
            for row in all_rows
            for family in str(row["method_families"]).split("|")
        }
    )
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
