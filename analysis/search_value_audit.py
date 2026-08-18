#!/usr/bin/env python3
"""Meeting-focused search-value audit for the fixed Checkpoint 4 probes."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import pearsonr, spearmanr


UNIFORM_LOG_LOSS = float(np.log(3.0))
VARIANTS = {
    "full": {"score": 1.0, "delta": 0.4, "novelty": 0.25},
    "score_only": {"score": 1.0, "delta": 0.0, "novelty": 0.0},
    "delta_only": {"score": 0.0, "delta": 1.0, "novelty": 0.0},
    "novelty_only": {"score": 0.0, "delta": 0.0, "novelty": 1.0},
    "score_delta": {"score": 1.0, "delta": 0.4, "novelty": 0.0},
    "score_novelty": {"score": 1.0, "delta": 0.0, "novelty": 0.25},
    "delta_novelty": {"score": 0.0, "delta": 0.4, "novelty": 0.25},
}

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
    result: set[str] = set()
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
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
        pass
    return result


def token_jaccard_distance(left: str, right: str) -> float:
    left_tokens = normalized_token_set(left)
    right_tokens = normalized_token_set(right)
    union = left_tokens | right_tokens
    return 0.0 if not union else 1.0 - len(left_tokens & right_tokens) / len(union)


def correlation(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame.x.nunique() < 2 or frame.y.nunique() < 2:
        return {"n": int(len(frame)), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    pearson = pearsonr(frame.x, frame.y)
    spearman = spearmanr(frame.x, frame.y)
    return {
        "n": int(len(frame)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def tied_ids(frame: pd.DataFrame, column: str, mode: str) -> set[str]:
    values = frame[column].dropna()
    target = values.min() if mode == "min" else values.max()
    return set(frame.loc[np.isclose(frame[column], target, equal_nan=False), "parent_id"])


def mean_for_ids(frame: pd.DataFrame, ids: set[str], column: str) -> float:
    values = frame.loc[frame.parent_id.isin(ids), column].dropna()
    return float(values.mean()) if len(values) else np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    probe_dir = args.probe_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    children = pd.DataFrame(
        json.loads(line)
        for line in (probe_dir / "probe_results.jsonl").read_text().splitlines()
        if line.strip()
    ).sort_values(["snapshot_step", "parent_utility", "repeat"], ascending=[True, False, True])
    if len(children) != 24 or children.probe_id.nunique() != 24:
        raise RuntimeError("Expected exactly 24 unique counterfactual children")

    manifest = json.loads((probe_dir / "probe_manifest.json").read_text())
    journal_path = Path(manifest["baseline_checkpoint"]) / "journal.jsonl"
    journal = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
    baseline_code = {str(node["id"]): str(node.get("code") or "") for node in journal}
    valid_baseline = [node for node in journal if node.get("metric") is not None and not node.get("is_buggy")]
    incumbent_node = min(valid_baseline, key=lambda node: float(node["metric"]))
    incumbent_id = str(incumbent_node["id"])
    incumbent_code = str(incumbent_node["code"])

    family_by_snapshot_parent: dict[tuple[int, str], str] = {}
    for snapshot in sorted(children.snapshot_step.unique()):
        data = json.loads((probe_dir / f"snapshot_{snapshot}.json").read_text())
        for candidate in data["candidate_utilities"]:
            family_by_snapshot_parent[(int(snapshot), candidate["node_id"])] = candidate["method_family_auto"]

    unit_rows: list[dict[str, object]] = []
    for (snapshot, parent_id), group in children.groupby(["snapshot_step", "parent_id"], sort=True):
        group = group.sort_values("repeat")
        first = group.iloc[0]
        valid_scores = group.child_score.dropna().astype(float)
        parent_fallback = group.child_score.fillna(first.parent_score).astype(float)
        uniform_penalty = group.child_score.fillna(UNIFORM_LOG_LOSS).astype(float)
        child_quality_gain = group.child_score.apply(
            lambda value: max(0.0, UNIFORM_LOG_LOSS - float(value)) if pd.notna(value) else 0.0
        )
        unit_rows.append(
            {
                "snapshot_step": int(snapshot),
                "parent_id": parent_id,
                "method_family": family_by_snapshot_parent[(int(snapshot), parent_id)],
                "parent_score": float(first.parent_score),
                "incumbent_score": float(first.snapshot_best_score),
                "full_utility": float(first.parent_utility),
                "selection_probability": float(first.parent_probability),
                "score_component": float(first.parent_score_component),
                "delta_component": float(first.parent_delta_component),
                "novelty_component": float(first.parent_novelty_component),
                "children": int(len(group)),
                "successful_children": int(valid_scores.count()),
                "failed_children": int(group.child_score.isna().sum()),
                "failure_rate": float(group.child_score.isna().mean()),
                "mean_child_score_success": float(valid_scores.mean()) if len(valid_scores) else np.nan,
                "best_child_score_success": float(valid_scores.min()) if len(valid_scores) else np.nan,
                "worst_child_score_success": float(valid_scores.max()) if len(valid_scores) else np.nan,
                "parent_fallback_mean_score": float(parent_fallback.mean()),
                "uniform_penalized_mean_score": float(uniform_penalty.mean()),
                "mean_incumbent_shortfall_success": float((valid_scores - first.snapshot_best_score).mean()) if len(valid_scores) else np.nan,
                "best_incumbent_shortfall": float((valid_scores - first.snapshot_best_score).min()) if len(valid_scores) else np.nan,
                "p_beat_incumbent": float((valid_scores < first.snapshot_best_score).sum() / len(group)),
                "expected_incumbent_gain": float(np.maximum(0.0, first.snapshot_best_score - valid_scores).sum() / len(group)),
                "mean_parent_relative_improvement": float(group.direction_corrected_improvement.mean()),
                "execution_seconds": float(group.execution_runtime.sum()),
                "wall_seconds": float(group.wall_runtime.sum()),
                "quality_gain_vs_uniform_per_execution_second": float(child_quality_gain.sum() / group.execution_runtime.sum()),
            }
        )
    units = pd.DataFrame(unit_rows).sort_values(["snapshot_step", "full_utility"], ascending=[True, False])

    for name, weights in VARIANTS.items():
        units[f"selector_{name}"] = (
            weights["score"] * units.score_component
            + weights["delta"] * units.delta_component
            + weights["novelty"] * units.novelty_component
        )

    selector_rows: list[dict[str, object]] = []
    oracle_rows: list[dict[str, object]] = []
    for snapshot, frame in units.groupby("snapshot_step", sort=True):
        frame = frame.copy()
        conditional_oracle = tied_ids(frame, "mean_child_score_success", "min")
        best2_oracle = tied_ids(frame, "best_child_score_success", "min")
        fallback_oracle = tied_ids(frame, "parent_fallback_mean_score", "min")
        uniform_oracle = tied_ids(frame, "uniform_penalized_mean_score", "min")
        oracle_rows.append(
            {
                "snapshot_step": int(snapshot),
                "conditional_mean_oracle": ";".join(sorted(conditional_oracle)),
                "best_of_two_oracle": ";".join(sorted(best2_oracle)),
                "parent_fallback_oracle": ";".join(sorted(fallback_oracle)),
                "uniform_penalty_oracle": ";".join(sorted(uniform_oracle)),
                "oracle_invariant_across_mean_failure_policies": bool(conditional_oracle == fallback_oracle == uniform_oracle),
            }
        )
        for variant in VARIANTS:
            selector_column = f"selector_{variant}"
            top_ids = tied_ids(frame, selector_column, "max")
            probabilities = softmax(frame[selector_column].to_numpy(dtype=float))
            conditional_top = mean_for_ids(frame, top_ids, "mean_child_score_success")
            best2_top = mean_for_ids(frame, top_ids, "best_child_score_success")
            fallback_top = mean_for_ids(frame, top_ids, "parent_fallback_mean_score")
            uniform_top = mean_for_ids(frame, top_ids, "uniform_penalized_mean_score")
            conditional_best = float(frame.mean_child_score_success.min())
            best2_best = float(frame.best_child_score_success.min())
            fallback_best = float(frame.parent_fallback_mean_score.min())
            uniform_best = float(frame.uniform_penalized_mean_score.min())
            selector_rows.append(
                {
                    "snapshot_step": int(snapshot),
                    "variant": variant,
                    "top_parent_ids": ";".join(sorted(top_ids)),
                    "conditional_mean_top1_agreement": bool(top_ids & conditional_oracle),
                    "conditional_mean_top1_regret": conditional_top - conditional_best,
                    "best_of_two_top1_agreement": bool(top_ids & best2_oracle),
                    "best_of_two_top1_regret": best2_top - best2_best,
                    "parent_fallback_top1_agreement": bool(top_ids & fallback_oracle),
                    "parent_fallback_top1_regret": fallback_top - fallback_best,
                    "uniform_penalty_top1_agreement": bool(top_ids & uniform_oracle),
                    "uniform_penalty_top1_regret": uniform_top - uniform_best,
                    "top_parent_failure_rate": mean_for_ids(frame, top_ids, "failure_rate"),
                    "policy_expected_parent_fallback_score": float(np.dot(probabilities, frame.parent_fallback_mean_score)),
                    "policy_parent_fallback_regret": float(np.dot(probabilities, frame.parent_fallback_mean_score) - fallback_best),
                    "policy_mass_on_fallback_oracle": float(probabilities[frame.parent_id.isin(fallback_oracle)].sum()),
                }
            )
    selector_snapshot = pd.DataFrame(selector_rows)
    oracles = pd.DataFrame(oracle_rows)
    selector_summary = (
        selector_snapshot.groupby("variant", as_index=False)
        .agg(
            conditional_mean_top1_rate=("conditional_mean_top1_agreement", "mean"),
            conditional_mean_regret=("conditional_mean_top1_regret", "mean"),
            best_of_two_top1_rate=("best_of_two_top1_agreement", "mean"),
            best_of_two_regret=("best_of_two_top1_regret", "mean"),
            parent_fallback_top1_rate=("parent_fallback_top1_agreement", "mean"),
            parent_fallback_regret=("parent_fallback_top1_regret", "mean"),
            uniform_penalty_top1_rate=("uniform_penalty_top1_agreement", "mean"),
            uniform_penalty_regret=("uniform_penalty_top1_regret", "mean"),
            mean_top_parent_failure_rate=("top_parent_failure_rate", "mean"),
            mean_policy_parent_fallback_regret=("policy_parent_fallback_regret", "mean"),
            mean_policy_mass_on_oracle=("policy_mass_on_fallback_oracle", "mean"),
        )
    )
    selector_summary["order"] = selector_summary.variant.map({name: i for i, name in enumerate(VARIANTS)})
    selector_summary = selector_summary.sort_values("order").drop(columns="order")

    signal_columns = {
        "full_utility": "full_utility",
        "score_component": "score_component",
        "delta_component": "delta_component",
        "novelty_component": "novelty_component",
        "parent_score": "parent_score",
    }
    outcome_columns = {
        "conditional_mean_child_score": "mean_child_score_success",
        "parent_fallback_mean_score": "parent_fallback_mean_score",
        "best_child_score": "best_child_score_success",
        "failure_rate": "failure_rate",
        "incumbent_shortfall": "mean_incumbent_shortfall_success",
    }
    correlation_rows: list[dict[str, object]] = []
    for signal_name, signal_column in signal_columns.items():
        for outcome_name, outcome_column in outcome_columns.items():
            result = correlation(units[signal_column], units[outcome_column])
            correlation_rows.append({"signal": signal_name, "outcome": outcome_name, **result})
    correlations = pd.DataFrame(correlation_rows)

    mechanism_rows: list[dict[str, object]] = []
    for child in children.itertuples():
        code = Path(child.code_path).read_text(encoding="utf-8")
        parent_code = baseline_code[child.parent_id]
        mechanism_rows.append(
            {
                "probe_id": child.probe_id,
                "snapshot_step": int(child.snapshot_step),
                "parent_id": child.parent_id,
                "method_family": family_by_snapshot_parent[(int(child.snapshot_step), child.parent_id)],
                "child_score": child.child_score,
                "status": child.status,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "distance_to_parent": token_jaccard_distance(code, parent_code),
                "distance_to_incumbent": token_jaccard_distance(code, incumbent_code),
                "uses_logistic_regression": "LogisticRegression" in code,
                "uses_linear_svc": "LinearSVC" in code,
                "uses_char_ngram_3_5": bool(re.search(r"ngram_range\s*=\s*\(\s*3\s*,\s*5\s*\)", code)),
                "uses_c_10": bool(re.search(r"\bC\s*=\s*10(?:\.0)?\b", code)),
                "uses_stratified_kfold": "StratifiedKFold" in code,
            }
        )
    mechanism = pd.DataFrame(mechanism_rows)
    focal = mechanism.loc[mechanism.parent_id.str.startswith("f28d7d")].copy()

    metrics = {
        "design": {
            "snapshots": int(units.snapshot_step.nunique()),
            "parent_snapshot_units": int(len(units)),
            "unique_parents": int(units.parent_id.nunique()),
            "children": int(len(children)),
            "successes": int(children.child_score.notna().sum()),
            "failures": int(children.child_score.isna().sum()),
            "incumbent_score": float(children.snapshot_best_score.iloc[0]),
        },
        "incumbent": {
            "children_beating_incumbent": int(children.child_new_global_best.sum()),
            "closest_child_score": float(children.child_score.min()),
            "closest_child_shortfall": float(children.child_score.min() - children.snapshot_best_score.iloc[0]),
        },
        "metric_robustness": {
            "snapshots_with_same_mean_quality_oracle_under_conditional_parent_fallback_and_uniform_penalty": int(oracles.oracle_invariant_across_mean_failure_policies.sum()),
            "total_snapshots": int(len(oracles)),
        },
        "focal_parent_mechanism": {
            "parent_id": str(focal.parent_id.iloc[0]),
            "incumbent_id": incumbent_id,
            "children": int(len(focal)),
            "distinct_code_hashes": int(focal.code_sha256.nunique()),
            "children_using_char_ngram_3_5": int(focal.uses_char_ngram_3_5.sum()),
            "children_using_c_10": int(focal.uses_c_10.sum()),
            "children_using_logistic_regression": int(focal.uses_logistic_regression.sum()),
            "children_using_linear_svc": int(focal.uses_linear_svc.sum()),
            "mean_distance_to_parent": float(focal.distance_to_parent.mean()),
            "mean_distance_to_incumbent": float(focal.distance_to_incumbent.mean()),
        },
        "correlations": {
            row.signal + "__" + row.outcome: {
                "n": int(row.n),
                "pearson_r": float(row.pearson_r),
                "pearson_p": float(row.pearson_p),
                "spearman_rho": float(row.spearman_rho),
                "spearman_p": float(row.spearman_p),
            }
            for row in correlations.itertuples()
        },
    }

    units.to_csv(output / "parent_snapshot_search_value.csv", index=False)
    selector_snapshot.to_csv(output / "selector_snapshot_comparison.csv", index=False)
    selector_summary.to_csv(output / "selector_ablation_summary.csv", index=False)
    correlations.to_csv(output / "signal_outcome_correlations.csv", index=False)
    oracles.to_csv(output / "outcome_oracle_sensitivity.csv", index=False)
    mechanism.to_csv(output / "child_mechanism_features.csv", index=False)
    (output / "analysis_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")

    # Figure 1: direct within-snapshot comparison, preserving the actual scale.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7), sharey=True)
    for ax, (snapshot, frame) in zip(axes, units.groupby("snapshot_step", sort=True)):
        frame = frame.sort_values("full_utility", ascending=False).reset_index(drop=True)
        full_top = tied_ids(frame, "full_utility", "max")
        oracle = tied_ids(frame, "mean_child_score_success", "min")
        colors = ["#54a24b" if pid in oracle else "#f58518" if pid in full_top else "#4c78a8" for pid in frame.parent_id]
        heights = frame.mean_child_score_success.fillna(0.0)
        colors = ["#d62728" if successes == 0 else color for successes, color in zip(frame.successful_children, colors)]
        bars = ax.bar(range(len(frame)), heights, color=colors)
        for bar, successes in zip(bars, frame.successful_children):
            if successes == 0:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.025, "0/2\nfailed", ha="center", va="bottom", fontsize=8, color="#a50f15")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{successes}/2", ha="center", va="bottom", fontsize=8)
        ax.axhline(frame.incumbent_score.iloc[0], color="black", linestyle="--", linewidth=1.2, label="incumbent")
        ax.set_xticks(range(len(frame)), [str(value)[:6] for value in frame.parent_id], rotation=35, ha="right")
        ax.set_title(f"Snapshot {snapshot}")
        ax.set_xlabel("Parent (full-utility rank)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean successful child log loss (lower is better)")
    fig.suptitle("Absolute descendant quality: green = oracle, orange = Frontis utility top-1")
    fig.tight_layout()
    fig.savefig(output / "absolute_child_quality_by_snapshot.png", dpi=180)
    plt.close(fig)

    # Figure 2: selector ablations and robustness to failure semantics.
    plot_summary = selector_summary.sort_values("conditional_mean_regret", ascending=True)
    y = np.arange(len(plot_summary))
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.barh(y - 0.18, plot_summary.conditional_mean_regret, height=0.34, label="conditional success mean")
    ax.barh(y + 0.18, plot_summary.parent_fallback_regret, height=0.34, label="failure → parent score")
    ax.set_yticks(y, plot_summary.variant)
    ax.invert_yaxis()
    ax.set_xlabel("Mean top-1 selection regret (log loss; lower is better)")
    ax.set_title("Component ablation on the fixed probed-parent support")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "selector_component_ablation.png", dpi=180)
    plt.close(fig)

    # Figure 3: direct signal validity. A negative slope is desirable because
    # higher signal should predict lower child loss.
    fig, axes = plt.subplots(1, 4, figsize=(14.5, 4.2), sharey=True)
    signal_plot = ["full_utility", "score_component", "delta_component", "novelty_component"]
    snapshot_colors = {4: "#4c78a8", 5: "#f58518", 9: "#54a24b"}
    for ax, signal in zip(axes, signal_plot):
        for snapshot, frame in units.groupby("snapshot_step"):
            ax.scatter(frame[signal], frame.parent_fallback_mean_score, s=65, color=snapshot_colors[int(snapshot)], label=f"snapshot {snapshot}")
        result = correlation(units[signal], units.parent_fallback_mean_score)
        ax.set_title(f"{signal.replace('_', ' ')}\nSpearman ρ={result['spearman_rho']:.3f}")
        ax.set_xlabel("Signal value")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean child score, failure → parent score")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Signal validity (negative correlation is desirable)")
    fig.tight_layout()
    fig.savefig(output / "utility_component_validity.png", dpi=180)
    plt.close(fig)

    full_corr = correlation(units.full_utility, units.parent_fallback_mean_score)
    novelty_corr = correlation(units.novelty_component, units.parent_fallback_mean_score)
    full_summary = selector_summary.set_index("variant").loc["full"]
    novelty_summary = selector_summary.set_index("variant").loc["novelty_only"]
    report = f"""# Frontis search-value validity audit

## Research question

The earlier counterfactual analysis measured parent-relative improvement, which can favor weak parents. This audit asks the decision-relevant question: **which probed parent produced the strongest absolute descendants under the same one-step `Improve` operator?**

The primary outcome is mean successful child log loss. Failures are reported separately, and robustness is checked by substituting either the parent score (no usable offspring) or uniform-prediction log loss (`ln(3)`) for failed runs.

## Main observations

1. **The parent-relative metric confound does not explain away the misranking.** The same ensemble parent (`f28d7d…`) produced the best mean absolute child score in all three snapshots. The mean-quality oracle was unchanged under conditional-success, parent-fallback, and uniform-penalty failure policies in {int(oracles.oracle_invariant_across_mean_failure_policies.sum())}/{len(oracles)} snapshots.
2. **Full Frontis utility did not rank one-step value on this support.** Utility versus parent-fallback child score has Spearman ρ={full_corr['spearman_rho']:.4f} (p={full_corr['spearman_p']:.4f}); because lower child score is better, a positive correlation is undesirable, but the estimate is weak and uncertain. More concretely, full utility's top-1 mean-quality agreement was {full_summary.conditional_mean_top1_rate:.0%}, with mean regret {full_summary.conditional_mean_regret:.5f}.
3. **The evidence does not implicate novelty as the harmful component.** Novelty versus parent-fallback child score has ρ={novelty_corr['spearman_rho']:.4f} (p={novelty_corr['spearman_p']:.4f}). Novelty-only ranked the mean-quality oracle first in {novelty_summary.conditional_mean_top1_rate:.0%} of snapshots. Under the unchanged softmax, however, it assigned only {novelty_summary.mean_policy_mass_on_oracle:.1%} mean probability mass to the oracle and reduced probability-weighted regret only modestly ({full_summary.mean_policy_parent_fallback_regret:.5f} → {novelty_summary.mean_policy_parent_fallback_regret:.5f}).
4. **The incumbent-improvement outcome is censored.** No child beat the incumbent ({children.snapshot_best_score.iloc[0]:.5f}); the closest child scored {children.child_score.min():.5f}, a shortfall of {children.child_score.min() - children.snapshot_best_score.iloc[0]:.5f}. Absolute child quality and shortfall therefore carry the useful resolution.
5. **This is a repeated-parent result, not three independent family discoveries.** The apparent novelty-only success is driven by the same ensemble parent across snapshots. It supports a targeted confirmatory rollout, not replacing Frontis with a novelty-only selector.
6. **The offspring reveal a plausible mechanism.** All {len(focal)} focal-parent children are code-distinct. The weak ensemble parent used character n-grams 4–6 and `C=1`; {int(focal.uses_char_ngram_3_5.sum())}/{len(focal)} descendants switched to the incumbent's 3–5 character n-grams and {int(focal.uses_c_10.sum())}/{len(focal)} adopted `C=10`. Thus the branch's value looks like memory-guided repair toward a known strong configuration, rather than six independent ensemble innovations.

## Selector ablation summary

All variants retain Frontis's original component scaling; an omitted component receives zero weight. Evaluation is restricted to the four probed parents in each snapshot.

{selector_summary.to_markdown(index=False)}

## Snapshot-level search value

{units[['snapshot_step','parent_id','method_family','parent_score','full_utility','score_component','delta_component','novelty_component','successful_children','mean_child_score_success','best_child_score_success','parent_fallback_mean_score','mean_incumbent_shortfall_success','failure_rate','execution_seconds']].to_markdown(index=False)}

## Conclusion for the meeting

The corrected diagnosis is more specific than “Frontis rewards weak parents.” In this case study, Frontis's fixed additive ranking overvalued current parent quality/progress relative to **repairability under the `Improve` prompt and search memory**: a weak cross-validation scaffold could absorb the incumbent's successful hyperparameters, while forcing changes to already-strong parents often regressed. Coarse family novelty happened to rank that scaffold highly, but the evidence does not show that its family label caused the value.

This provides credible case-study evidence of a one-step ranking failure mechanism, while leaving open whether the repaired branch has superior multi-step value. The next justified experiment is a paired, equal-budget two-step rollout of the Frontis utility winner versus `f28d7d…`, not an immediate global selector rewrite.

## Limits

- Three snapshots, four probed parents per snapshot, and two offspring per parent.
- The same ensemble parent appears in all snapshots.
- Component ablations are retrospective and restricted to probed support; unprobed candidates have unknown outcomes.
- No child beat the incumbent, so global-best probability cannot discriminate parents here.
- No learned selector was fitted and no baseline trajectory was modified.
"""
    (output / "LAB_MEETING_FRONTIS_SEARCH_VALUE.md").write_text(report)
    print(json.dumps(metrics, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
