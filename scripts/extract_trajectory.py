#!/usr/bin/env python3
"""Normalize one OpenMLE-Evo task trajectory without inventing missing fields."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = [
    "task",
    "run_id",
    "step_id",
    "node_id",
    "parent_id",
    "parent_2_id",
    "operator",
    "score",
    "delta",
    "novelty",
    "method_family",
    "family_fail_rate",
    "selection_utility",
    "selection_probability",
    "status",
    "runtime",
    "prompt_tokens",
    "completion_tokens",
    "error_signature",
    "code_path",
    "response_path",
]

EXTRA_COLUMNS = [
    "parent_2_selection_utility",
    "parent_2_selection_probability",
    "model_runtime",
    "model_plus_sandbox_runtime",
    "total_tokens",
    "reasoning_path",
    "feedback_path",
    "clear_log_path",
    "raw_log_path",
    "experience_card_path",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def find_task_root(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("program_ep_*/**/experience_cards.jsonl"))
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected one task trajectory below {run_dir}, found {len(candidates)}"
        )
    return candidates[0].parent


def selected_parent_value(
    card: dict[str, Any], parent_id: str | None, field: str
) -> Any:
    if parent_id is None:
        return None
    for item in card.get("selection_utility") or []:
        if item.get("node_id") == parent_id:
            return item.get(field)
    return None


def existing_path(path: Path) -> str | None:
    return str(path.resolve()) if path.is_file() else None


def build_rows(run_dir: Path, task_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_stat = read_json(task_root / "stat.json")
    board = read_json(task_root / "strategy_board.json")
    family_stats = board.get("method_family_stats", {})
    rows: list[dict[str, Any]] = []

    for step in task_stat.get("steps", []):
        step_id = step["step"]
        step_dir = task_root / f"step_{step_id}"
        card = read_json(step_dir / "experience_card.json")
        parent_ids = card.get("parent_node_ids") or []
        # Draft cards point at a synthetic root, while their persisted `parents` list is
        # empty. Preserve actual search lineage and do not report the synthetic root.
        if not card.get("parents"):
            parent_ids = []
        parent_id = parent_ids[0] if parent_ids else None
        parent_2_id = parent_ids[1] if len(parent_ids) > 1 else None
        method_family = card.get("method_family_auto")
        row = {
            "task": task_stat.get("task_name"),
            "run_id": run_dir.name,
            "step_id": card.get("step_id", step_id),
            "node_id": card.get("node_id", step.get("node_id")),
            "parent_id": parent_id,
            "parent_2_id": parent_2_id,
            "operator": card.get("operator", step.get("operator")),
            "score": card.get("score", step.get("score")),
            "delta": card.get("delta_vs_parent"),
            "novelty": card.get("novelty_score"),
            "method_family": method_family,
            "family_fail_rate": (family_stats.get(method_family) or {}).get("fail_rate"),
            "selection_utility": selected_parent_value(card, parent_id, "utility"),
            "selection_probability": selected_parent_value(card, parent_id, "probability"),
            "status": card.get("status", step.get("status")),
            "runtime": card.get("sandbox_time_used", step.get("sandbox_time_used")),
            "prompt_tokens": card.get("prompt_tokens", step.get("prompt_tokens")),
            "completion_tokens": card.get(
                "completion_tokens", step.get("completion_tokens")
            ),
            "error_signature": card.get("error_signature"),
            "code_path": existing_path(step_dir / "valid_code.py"),
            "response_path": existing_path(step_dir / "response.md"),
            "parent_2_selection_utility": selected_parent_value(
                card, parent_2_id, "utility"
            ),
            "parent_2_selection_probability": selected_parent_value(
                card, parent_2_id, "probability"
            ),
            "model_runtime": card.get("model_time_used"),
            "model_plus_sandbox_runtime": card.get("model_plus_sandbox_time_used"),
            "total_tokens": card.get("total_tokens", step.get("total_tokens")),
            "reasoning_path": existing_path(step_dir / "reasoning_content.md"),
            "feedback_path": existing_path(step_dir / "feedback.txt"),
            "clear_log_path": existing_path(step_dir / "clear_run_log.txt"),
            "raw_log_path": existing_path(step_dir / "raw_run_log.txt"),
            "experience_card_path": existing_path(step_dir / "experience_card.json"),
        }
        rows.append(row)

    return rows, task_stat


def summarize(
    rows: list[dict[str, Any]], task_stat: dict[str, Any], task_root: Path
) -> dict[str, Any]:
    valid = [row for row in rows if row["status"] == "success"]
    invalid = [row for row in rows if row["status"] != "success"]
    scores = [row["score"] for row in valid if row["score"] is not None]
    parent_counts = Counter(
        parent
        for row in rows
        for parent in (row["parent_id"], row["parent_2_id"])
        if parent
    )
    error_counts = Counter(
        row["error_signature"] for row in invalid if row["error_signature"]
    )
    return {
        "task": task_stat.get("task_name"),
        "node_count": len(rows),
        "valid_count": len(valid),
        "invalid_count": len(invalid),
        "operator_counts": dict(Counter(row["operator"] for row in rows)),
        "score_progression": [
            {"step_id": row["step_id"], "score": row["score"]} for row in rows
        ],
        "best_validation_score": min(scores) if scores else None,
        "best_node_id": min(valid, key=lambda row: row["score"])["node_id"]
        if scores
        else None,
        "final_submit_score": task_stat.get("submit_score"),
        "parent_selection_counts": dict(parent_counts),
        "repeated_error_signatures": dict(error_counts),
        "failed_nodes_without_stored_error_signature": [
            row["node_id"] for row in invalid if not row["error_signature"]
        ],
        "observations": {
            "novelty_nonincreasing": all(
                rows[index]["novelty"] <= rows[index - 1]["novelty"]
                for index in range(1, len(rows))
                if rows[index]["novelty"] is not None
                and rows[index - 1]["novelty"] is not None
            ),
            "debug_repairs": [
                {
                    "node_id": row["node_id"],
                    "parent_id": row["parent_id"],
                    "score": row["score"],
                }
                for row in rows
                if row["operator"] == "debug"
            ],
        },
        "retained_artifacts": {
            "task_stat": str((task_root / "stat.json").resolve()),
            "experience_cards": str((task_root / "experience_cards.jsonl").resolve()),
            "strategy_board": str((task_root / "strategy_board.json").resolve()),
        },
    }


def write_outputs(
    rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = REQUIRED_COLUMNS + EXTRA_COLUMNS
    with (output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "trajectory.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output_dir / "trajectory_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate(rows: list[dict[str, Any]]) -> None:
    node_ids = {row["node_id"] for row in rows}
    if len(node_ids) != len(rows):
        raise SystemExit("Duplicate node_id detected")
    for row in rows:
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            raise SystemExit(f"Row {row['step_id']} lacks columns: {missing}")
        for parent_key in ("parent_id", "parent_2_id"):
            parent = row[parent_key]
            if parent and parent not in node_ids:
                raise SystemExit(f"Unknown {parent_key} {parent} at step {row['step_id']}")
        for path_key in (
            "code_path",
            "response_path",
            "reasoning_path",
            "feedback_path",
            "clear_log_path",
            "raw_log_path",
            "experience_card_path",
        ):
            path = row[path_key]
            if path and not Path(path).is_file():
                raise SystemExit(f"Missing retained artifact: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    task_root = find_task_root(run_dir)
    output_dir = (args.output_dir or run_dir / "trajectory_extract").resolve()
    rows, task_stat = build_rows(run_dir, task_root)
    validate(rows)
    summary = summarize(rows, task_stat, task_root)
    write_outputs(rows, summary, output_dir)
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
