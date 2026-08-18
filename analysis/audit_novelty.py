#!/usr/bin/env python3
"""Checkpoint 5: audit Frontis family-rarity novelty against code distance."""

from __future__ import annotations

import argparse
import io
import json
import tokenize
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, pearsonr, spearmanr


IGNORED_TOKEN_TYPES = {
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.NEWLINE,
    tokenize.NL,
    tokenize.COMMENT,
}


def normalized_token_set(code: str) -> set[str]:
    """Return a stable lexical-token set with literals normalized by type."""
    result: set[str] = set()
    try:
        stream = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in stream:
            if token.type in IGNORED_TOKEN_TYPES:
                continue
            if token.type == tokenize.STRING:
                result.add("STRING")
            elif token.type == tokenize.NUMBER:
                result.add("NUMBER")
            elif token.type == tokenize.NAME:
                result.add(f"NAME:{token.string}")
            elif token.type == tokenize.OP:
                result.add(f"OP:{token.string}")
            else:
                result.add(f"T{token.type}:{token.string}")
    except (IndentationError, SyntaxError, tokenize.TokenError):
        # Tokenization still yields the valid prefix, which is preferable to
        # dropping failed generated programs from the failure audit.
        pass
    return result


def jaccard_distance(left: set[str], right: set[str]) -> float:
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def corr(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    clean = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(clean) < 3 or clean.x.nunique() < 2 or clean.y.nunique() < 2:
        return {"n": int(len(clean)), "pearson_r": float("nan"), "pearson_p": float("nan"), "spearman_rho": float("nan"), "spearman_p": float("nan")}
    pearson = pearsonr(clean.x, clean.y)
    spearman = spearmanr(clean.x, clean.y)
    return {
        "n": int(len(clean)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--probe-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(args.trajectory).sort_values("step_id").reset_index(drop=True)
    if len(nodes) != 10 or nodes.node_id.nunique() != 10:
        raise RuntimeError(f"Expected 10 unique baseline nodes; found {len(nodes)}")

    code_by_id: dict[str, str] = {}
    tokens_by_id: dict[str, set[str]] = {}
    for row in nodes.itertuples():
        code = Path(row.code_path).read_text(encoding="utf-8")
        code_by_id[row.node_id] = code
        tokens_by_id[row.node_id] = normalized_token_set(code)

    node_rows: list[dict[str, object]] = []
    for index, row in nodes.iterrows():
        node_id = row.node_id
        prior_ids = nodes.loc[: index - 1, "node_id"].tolist() if index else []
        parent_ids = [value for value in (row.parent_id, row.parent_2_id) if pd.notna(value)]
        prior_distances = [jaccard_distance(tokens_by_id[node_id], tokens_by_id[other]) for other in prior_ids]
        parent_distances = [jaccard_distance(tokens_by_id[node_id], tokens_by_id[parent]) for parent in parent_ids]
        node_rows.append(
            {
                "step_id": int(row.step_id),
                "node_id": node_id,
                "operator": row.operator,
                "method_family": row.method_family,
                "frontis_novelty": float(row.novelty),
                "status": row.status,
                "failed": row.status != "success",
                "token_set_size": len(tokens_by_id[node_id]),
                "parent_count": len(parent_ids),
                "closest_parent_distance": min(parent_distances) if parent_distances else np.nan,
                "mean_parent_distance": float(np.mean(parent_distances)) if parent_distances else np.nan,
                "nearest_prior_distance": min(prior_distances) if prior_distances else np.nan,
                "mean_prior_distance": float(np.mean(prior_distances)) if prior_distances else np.nan,
                "family_fail_rate": float(row.family_fail_rate),
            }
        )
    node_metrics = pd.DataFrame(node_rows)

    pair_rows: list[dict[str, object]] = []
    node_lookup = nodes.set_index("node_id")
    for left, right in combinations(nodes.node_id, 2):
        left_family = node_lookup.loc[left, "method_family"]
        right_family = node_lookup.loc[right, "method_family"]
        pair_rows.append(
            {
                "left_id": left,
                "right_id": right,
                "left_family": left_family,
                "right_family": right_family,
                "same_family": left_family == right_family,
                "token_jaccard_distance": jaccard_distance(tokens_by_id[left], tokens_by_id[right]),
            }
        )
    pairs = pd.DataFrame(pair_rows)

    within = pairs.loc[pairs.same_family, "token_jaccard_distance"]
    between = pairs.loc[~pairs.same_family, "token_jaccard_distance"]
    family_test = mannwhitneyu(between, within, alternative="two-sided")

    novelty_threshold = float(node_metrics.frontis_novelty.median())
    node_metrics["high_novelty"] = node_metrics.frontis_novelty >= novelty_threshold
    failure_table = pd.crosstab(node_metrics.high_novelty, node_metrics.failed).reindex(index=[True, False], columns=[True, False], fill_value=0)
    fisher = fisher_exact(failure_table.to_numpy(), alternative="two-sided")

    probe_rows = [json.loads(line) for line in args.probe_results.read_text().splitlines() if line.strip()]
    probes = pd.DataFrame(probe_rows)
    parent_family = nodes.set_index("node_id")[["method_family", "family_fail_rate"]]
    probe_parent = (
        probes.groupby("parent_id", as_index=False)
        .agg(
            probe_children=("probe_id", "count"),
            snapshots=("snapshot_step", "nunique"),
            mean_downstream_improvement=("direction_corrected_improvement", "mean"),
            best_downstream_improvement=("direction_corrected_improvement", "max"),
            downstream_improve_rate=("child_improves_parent", "mean"),
            downstream_failure_rate=("status", lambda s: float((s != "success").mean())),
        )
        .join(parent_family, on="parent_id", validate="one_to_one")
    )

    novelty_correlations = {
        "closest_parent_distance": corr(node_metrics.frontis_novelty, node_metrics.closest_parent_distance),
        "nearest_prior_distance": corr(node_metrics.frontis_novelty, node_metrics.nearest_prior_distance),
        "mean_prior_distance": corr(node_metrics.frontis_novelty, node_metrics.mean_prior_distance),
    }
    novelty_failure_corr = corr(node_metrics.frontis_novelty, node_metrics.failed.astype(int))
    family_history_corr = corr(probe_parent.family_fail_rate, probe_parent.mean_downstream_improvement)

    family_summary = (
        node_metrics.groupby("method_family", as_index=False)
        .agg(
            nodes=("node_id", "count"),
            failures=("failed", "sum"),
            observed_failure_rate=("failed", "mean"),
            mean_frontis_novelty=("frontis_novelty", "mean"),
            mean_nearest_prior_distance=("nearest_prior_distance", "mean"),
        )
    )
    family_probe_summary = (
        probe_parent.groupby("method_family", as_index=False)
        .agg(
            unique_parents=("parent_id", "count"),
            probe_children=("probe_children", "sum"),
            stored_family_fail_rate=("family_fail_rate", "first"),
            mean_parent_downstream_improvement=("mean_downstream_improvement", "mean"),
            mean_parent_improve_rate=("downstream_improve_rate", "mean"),
        )
    )

    metrics = {
        "design": {
            "baseline_nodes": int(len(node_metrics)),
            "failed_nodes": int(node_metrics.failed.sum()),
            "pairwise_code_comparisons": int(len(pairs)),
            "families": node_metrics.method_family.value_counts().to_dict(),
            "distance": "1 - Jaccard(normalized Python lexical-token sets)",
        },
        "novelty_vs_code_distance": novelty_correlations,
        "family_distinctness": {
            "within_family_pairs": int(len(within)),
            "between_family_pairs": int(len(between)),
            "within_family_mean_distance": float(within.mean()),
            "between_family_mean_distance": float(between.mean()),
            "between_minus_within": float(between.mean() - within.mean()),
            "naive_pairwise_mann_whitney_u": float(family_test.statistic),
            "naive_pairwise_mann_whitney_p": float(family_test.pvalue),
            "caveat": "Only one ensemble node exists; all between-family pairs share that node and are not independent.",
        },
        "high_novelty_failure": {
            "median_threshold": novelty_threshold,
            "high_novelty_nodes": int(node_metrics.high_novelty.sum()),
            "high_novelty_failure_rate": float(node_metrics.loc[node_metrics.high_novelty, "failed"].mean()),
            "lower_novelty_failure_rate": float(node_metrics.loc[~node_metrics.high_novelty, "failed"].mean()),
            "fisher_odds_ratio": float(fisher.statistic),
            "fisher_p": float(fisher.pvalue),
            "novelty_vs_failure_correlation": novelty_failure_corr,
        },
        "family_failure_history_vs_downstream_improvement": {
            "unit": "unique Checkpoint 4 parent, with all its probe children aggregated",
            "correlation": family_history_corr,
            "caveat": "Only two families and one ensemble parent are represented; stored family_fail_rate is static in the extracted trajectory, not reconstructed at each historical selection step.",
        },
    }

    node_metrics.to_csv(output / "node_novelty_distance.csv", index=False)
    pairs.to_csv(output / "pairwise_code_distance.csv", index=False)
    family_summary.to_csv(output / "family_summary.csv", index=False)
    probe_parent.to_csv(output / "probe_parent_family_outcomes.csv", index=False)
    family_probe_summary.to_csv(output / "family_probe_summary.csv", index=False)
    (output / "analysis_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    colors = {"sklearn": "#4c78a8", "ensemble": "#f58518"}
    markers = {False: "o", True: "X"}
    plot_data = node_metrics.dropna(subset=["nearest_prior_distance"])
    for (family, failed), group in plot_data.groupby(["method_family", "failed"]):
        axes[0].scatter(group.frontis_novelty, group.nearest_prior_distance, s=85, color=colors.get(family, "gray"), marker=markers[bool(failed)], label=f"{family}, {'failed' if failed else 'success'}")
    rho = novelty_correlations["nearest_prior_distance"]["spearman_rho"]
    axes[0].set_title(f"Novelty vs nearest-prior distance (ρ={rho:.3f})")
    axes[0].set_xlabel("Frontis novelty")
    axes[0].set_ylabel("Normalized token-set Jaccard distance")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    rng = np.random.default_rng(42)
    for position, (label, values) in enumerate((("within family", within), ("between families", between))):
        x = position + rng.uniform(-0.08, 0.08, size=len(values))
        axes[1].scatter(x, values, alpha=0.65, s=35)
        axes[1].plot([position - 0.18, position + 0.18], [values.mean(), values.mean()], color="black", linewidth=2)
    axes[1].set_xticks([0, 1], ["within family", "between families"])
    axes[1].set_ylabel("Normalized token-set Jaccard distance")
    axes[1].set_title("Family label vs pairwise code distance")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "novelty_code_distance.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    failure_rates = [
        node_metrics.loc[node_metrics.high_novelty, "failed"].mean(),
        node_metrics.loc[~node_metrics.high_novelty, "failed"].mean(),
    ]
    axes[0].bar(["High novelty\n(≥ median)", "Lower novelty"], failure_rates, color=["#f58518", "#4c78a8"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Observed failure rate")
    axes[0].set_title("High-novelty branches did not fail more")
    axes[0].grid(axis="y", alpha=0.25)

    for family, group in probe_parent.groupby("method_family"):
        axes[1].scatter(group.family_fail_rate, group.mean_downstream_improvement, s=90, color=colors.get(family, "gray"), label=family)
    axes[1].set_xlabel("Stored family failure rate")
    axes[1].set_ylabel("Mean downstream positive improvement")
    axes[1].set_title("Family history vs Checkpoint 4 outcomes")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "failure_and_downstream.png", dpi=180)
    plt.close(fig)

    nearest = novelty_correlations["nearest_prior_distance"]
    report = f"""# Checkpoint 5 novelty audit

This audit compares Frontis family-rarity novelty with `1 − Jaccard` over normalized Python lexical-token sets. String and numeric literals are normalized, comments/layout tokens are removed, and generated failures remain included using any tokenizable prefix.

## Results

| Question | Statistic | Result |
|---|---|---:|
| Does novelty track nearest prior code distance? | Spearman ρ (n={nearest['n']}) | {nearest['spearman_rho']:.4f} (p={nearest['spearman_p']:.4f}) |
| Does novelty track closest-parent distance? | Spearman ρ (n={novelty_correlations['closest_parent_distance']['n']}) | {novelty_correlations['closest_parent_distance']['spearman_rho']:.4f} (p={novelty_correlations['closest_parent_distance']['spearman_p']:.4f}) |
| Are between-family programs farther apart? | Mean between − within | {between.mean() - within.mean():.4f} |
| Do high-novelty nodes fail more? | Failure rate, high vs lower | {node_metrics.loc[node_metrics.high_novelty, 'failed'].mean():.1%} vs {node_metrics.loc[~node_metrics.high_novelty, 'failed'].mean():.1%} |
| Does stored family failure rate predict downstream improvement? | Spearman ρ, unique parents (n={family_history_corr['n']}) | {family_history_corr['spearman_rho']:.4f} (p={family_history_corr['spearman_p']:.4f}) |

## Family distinctness

{family_summary.to_markdown(index=False)}

Across all code pairs, mean within-family distance is {within.mean():.4f}; mean between-family distance is {between.mean():.4f}. A naive pairwise Mann–Whitney calculation gives p={family_test.pvalue:.4f}, but it is not a valid independent-pairs test here: the trajectory contains nine `sklearn` nodes and only one `ensemble` node, so all nine between-family pairs share the same ensemble program. The distance difference is therefore descriptive only.

## High-novelty failure

Using the median novelty ({novelty_threshold:.4f}) as a descriptive split, high-novelty nodes failed at {node_metrics.loc[node_metrics.high_novelty, 'failed'].mean():.1%}, versus {node_metrics.loc[~node_metrics.high_novelty, 'failed'].mean():.1%} for lower-novelty nodes (two-sided Fisher p={fisher.pvalue:.4f}). This run does not support the hypothesis that high novelty itself caused more failures; both failures occurred later among lower-novelty `sklearn` nodes.

## Family failure history and downstream improvement

{family_probe_summary.to_markdown(index=False)}

The unique-parent correlation uses completed Checkpoint 4 offspring and aggregates repeated snapshots before analysis. The apparent relationship is not identifiable from family effects: only two families are present, `ensemble` has a single parent, and the stored failure-rate field is static rather than a time-indexed historical covariate.

## Interpretation

This tiny, highly imbalanced trajectory cannot validate family rarity as a general novelty measure. The audit indicates whether its rankings align with lexical program diversity in this run, while the family-level questions remain underpowered. No novelty formula or selector behavior was changed.
"""
    (output / "CHECKPOINT5_REPORT.md").write_text(report)
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
