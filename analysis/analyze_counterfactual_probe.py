#!/usr/bin/env python3
"""Summarize the fixed Checkpoint 4 counterfactual parent probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def correlation(x: pd.Series, y: pd.Series) -> dict[str, float]:
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    args = parser.parse_args()
    probe_dir = args.probe_dir.resolve()
    results_path = probe_dir / "probe_results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    children = pd.DataFrame(rows).sort_values(["snapshot_step", "parent_utility", "repeat"], ascending=[True, False, True])
    if len(children) != 24 or children["probe_id"].nunique() != 24:
        raise RuntimeError(f"Expected 24 unique probes, found {len(children)} rows / {children['probe_id'].nunique()} IDs")

    keys = ["snapshot_step", "parent_id"]
    parents = (
        children.groupby(keys, as_index=False)
        .agg(
            parent_step=("parent_step", "first"),
            parent_score=("parent_score", "first"),
            parent_utility=("parent_utility", "first"),
            parent_probability=("parent_probability", "first"),
            snapshot_best_score=("snapshot_best_score", "first"),
            probes=("probe_id", "count"),
            successful_children=("status", lambda s: int((s == "success").sum())),
            improved_children=("child_improves_parent", "sum"),
            global_best_children=("child_new_global_best", "sum"),
            observed_improvement_mean=("direction_corrected_improvement", "mean"),
            observed_improvement_best=("direction_corrected_improvement", "max"),
            handoff_formula_mean=("handoff_formula_positive_improvement", "mean"),
            execution_seconds=("execution_runtime", "sum"),
        )
        .sort_values(["snapshot_step", "parent_utility"], ascending=[True, False])
    )
    parents["improvement_per_execution_second"] = (
        parents["observed_improvement_mean"] * parents["probes"] / parents["execution_seconds"]
    )

    snapshot_rows: list[dict[str, object]] = []
    top1_hits = 0
    for snapshot, group in parents.groupby("snapshot_step", sort=True):
        utility_top = group.loc[group["parent_utility"].idxmax()]
        best_observed = float(group["observed_improvement_mean"].max())
        observed_winners = set(group.loc[np.isclose(group["observed_improvement_mean"], best_observed), "parent_id"])
        hit = utility_top["parent_id"] in observed_winners
        top1_hits += int(hit)
        snapshot_rows.append(
            {
                "snapshot_step": int(snapshot),
                "utility_top_parent": utility_top["parent_id"],
                "utility_top_utility": float(utility_top["parent_utility"]),
                "utility_top_mean_improvement": float(utility_top["observed_improvement_mean"]),
                "observed_best_parent_ids": ";".join(sorted(observed_winners)),
                "observed_best_mean_improvement": best_observed,
                "top1_agreement": bool(hit),
                "selection_regret": best_observed - float(utility_top["observed_improvement_mean"]),
            }
        )
    snapshots = pd.DataFrame(snapshot_rows)

    primary_corr = correlation(parents["parent_utility"], parents["observed_improvement_mean"])
    handoff_corr = correlation(parents["parent_utility"], parents["handoff_formula_mean"])
    total_improvement = float(children["direction_corrected_improvement"].sum())
    total_execution = float(children["execution_runtime"].sum())
    metrics = {
        "design": {
            "snapshots": int(children["snapshot_step"].nunique()),
            "parent_snapshot_units": int(len(parents)),
            "children": int(len(children)),
            "children_per_parent": sorted(children.groupby(keys).size().unique().tolist()),
        },
        "primary_minimization_corrected": {
            "improvement_definition": "max(0, parent_score - child_score)",
            "utility_vs_parent_mean_improvement": primary_corr,
            "top1_agreement_count": int(top1_hits),
            "top1_agreement_rate": float(top1_hits / len(snapshots)),
            "mean_selection_regret": float(snapshots["selection_regret"].mean()),
            "median_selection_regret": float(snapshots["selection_regret"].median()),
            "p_child_improves_parent": float(children["child_improves_parent"].mean()),
            "p_child_new_global_best": float(children["child_new_global_best"].mean()),
            "successful_child_rate": float((children["status"] == "success").mean()),
            "total_positive_improvement": total_improvement,
            "total_execution_seconds": total_execution,
            "improvement_per_execution_second": float(total_improvement / total_execution),
        },
        "literal_handoff_formula_diagnostic": {
            "formula": "max(0, child_score - parent_score)",
            "note": "This rewards higher log loss even though the task is minimized; retained only as a diagnostic.",
            "utility_vs_parent_mean_formula_value": handoff_corr,
            "mean_formula_value": float(children["handoff_formula_positive_improvement"].mean()),
        },
    }

    children.to_csv(probe_dir / "probe_results.csv", index=False)
    parents.to_csv(probe_dir / "parent_aggregate.csv", index=False)
    snapshots.to_csv(probe_dir / "snapshot_selection_metrics.csv", index=False)
    (probe_dir / "analysis_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    markers = {4: "o", 5: "s", 9: "^"}
    for snapshot, group in parents.groupby("snapshot_step"):
        ax.scatter(group["parent_utility"], group["observed_improvement_mean"], s=80, marker=markers[int(snapshot)], label=f"snapshot {snapshot}")
    ax.set_xlabel("Frontis parent utility")
    ax.set_ylabel("Mean positive improvement (lower log loss is better)")
    ax.set_title(f"Utility vs counterfactual improvement (Spearman ρ={primary_corr['spearman_rho']:.3f})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(probe_dir / "utility_vs_improvement.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, (snapshot, group) in zip(axes, parents.groupby("snapshot_step", sort=True)):
        group = group.sort_values("parent_utility", ascending=False).reset_index(drop=True)
        colors = ["#d95f02" if i == 0 else "#4c78a8" for i in range(len(group))]
        ax.bar(range(len(group)), group["observed_improvement_mean"], color=colors)
        ax.set_xticks(range(len(group)), [str(x)[:6] for x in group["parent_id"]], rotation=35, ha="right")
        ax.set_title(f"Snapshot {snapshot}")
        ax.set_xlabel("Parent (utility rank order)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean positive improvement")
    fig.suptitle("Observed improvement by probed parent (orange = utility top-1)")
    fig.tight_layout()
    fig.savefig(probe_dir / "snapshot_parent_improvement.png", dpi=180)
    plt.close(fig)

    report = f"""# Checkpoint 4 counterfactual parent probe

The probe contains {len(children)} children from {len(parents)} parent-snapshot units across {len(snapshots)} fixed snapshots. Each parent has two forced `Improve` offspring, and probe children were not committed to the baseline search tree.

## Primary results (log loss minimization)

| Statistic | Result |
|---|---:|
| Pearson utility/improvement correlation | {primary_corr['pearson_r']:.4f} (p={primary_corr['pearson_p']:.4f}) |
| Spearman utility/improvement correlation | {primary_corr['spearman_rho']:.4f} (p={primary_corr['spearman_p']:.4f}) |
| Top-1 agreement | {top1_hits}/{len(snapshots)} ({top1_hits / len(snapshots):.1%}) |
| Mean selection regret | {snapshots['selection_regret'].mean():.6f} |
| P(child improves parent) | {children['child_improves_parent'].mean():.1%} |
| P(child becomes new global best) | {children['child_new_global_best'].mean():.1%} |
| Successful evaluation rate | {(children['status'] == 'success').mean():.1%} |
| Positive improvement / execution-second | {total_improvement / total_execution:.8f} |

`positive_improvement` is direction-corrected to `max(0, parent_score - child_score)` because this task minimizes log loss. The handoff's literal `max(0, child_score - parent_score)` is also retained in the outputs as a diagnostic, but it rewards degradation and is not used for the scientific conclusions.

## Snapshot selection

{snapshots.to_markdown(index=False)}

## Parent aggregates

{parents[['snapshot_step','parent_id','parent_score','parent_utility','successful_children','improved_children','global_best_children','observed_improvement_mean','observed_improvement_best','execution_seconds']].to_markdown(index=False)}

## Interpretation boundary

This is a small diagnostic sample (12 parent-snapshot units), so correlation p-values are descriptive and no selector model is fitted. Conclusions should be treated as evidence about this run and task, not as a general estimate of Frontis selector quality.
"""
    (probe_dir / "CHECKPOINT4_REPORT.md").write_text(report)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
