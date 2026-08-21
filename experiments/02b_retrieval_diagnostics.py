"""Analyze an existing Gate 2 retrieval CSV without retraining or re-encoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from latent_stroke_dynamics.retrieval_diagnostics import (
    CANDIDATE_NAMES,
    merge_test_metadata,
    summarize_retrieval,
    summarize_retrieval_by,
    summarize_retrieval_families,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _model_order(summary: pd.DataFrame) -> list[str]:
    preferred = ["identity", "mean_delta", "linear", "mlp"]
    observed = list(summary["model"].unique())
    return [name for name in preferred if name in observed] + [
        name for name in observed if name not in preferred
    ]


def _plot_candidate_preferences(summary: pd.DataFrame, output_path: Path) -> None:
    order = _model_order(summary)
    bottom = np.zeros(len(order), dtype=float)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for candidate in CANDIDATE_NAMES:
        values = summary.set_index("model").loc[
            order,
            f"predicted_{candidate}_rate",
        ].to_numpy()
        axis.bar(order, values, bottom=bottom, label=candidate)
        bottom += values
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean fraction selected across model seeds")
    axis.set_title("Which counterfactual outcome each model retrieves")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_pairwise_wins(summary: pd.DataFrame, output_path: Path) -> None:
    order = _model_order(summary)
    alternatives = list(CANDIDATE_NAMES[1:])
    x = np.arange(len(order), dtype=float)
    width = 0.24
    indexed = summary.set_index("model")
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for index, alternative in enumerate(alternatives):
        values = indexed.loc[
            order,
            f"true_beats_{alternative}_rate",
        ].to_numpy()
        axis.bar(
            x + (index - 1) * width,
            values,
            width=width,
            label=alternative,
        )
    axis.axhline(0.5, color="black", linestyle=":", label="50% pairwise")
    axis.set_xticks(x, order)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean true-outcome win rate across seeds")
    axis.set_title("True outcome versus each counterfactual class")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_margins(retrieval: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    for model in ["linear", "mlp"]:
        values = retrieval.loc[retrieval["model"] == model, "true_margin"]
        if not values.empty:
            axis.hist(values, bins=24, alpha=0.55, label=model)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Best-counterfactual score minus true score")
    axis.set_ylabel("Model-seed predictions")
    axis.set_title("Counterfactual retrieval margins across seeds")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_by_crowding(frame: pd.DataFrame, output_path: Path) -> None:
    trainable = frame.loc[frame["model"].isin(["linear", "mlp"])]
    levels = sorted(trainable["crowding"].unique())
    models = [name for name in ["linear", "mlp"] if name in set(trainable["model"])]
    x = np.arange(len(levels), dtype=float)
    width = 0.34
    figure, axis = plt.subplots(figsize=(7, 4.8))
    for index, model in enumerate(models):
        grouped = (
            trainable.loc[trainable["model"] == model]
            .groupby("crowding")["top1_accuracy"]
            .mean()
        )
        values = [float(grouped.loc[level]) for level in levels]
        axis.bar(x + (index - 0.5) * width, values, width=width, label=model)
    axis.axhline(0.25, color="black", linestyle=":", label="25% chance")
    axis.axhline(0.50, color="tab:green", linestyle="--", label="50% threshold")
    axis.set_xticks(x, [str(level) for level in levels])
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Prior-stroke crowding")
    axis.set_ylabel("Mean counterfactual top-1 accuracy across seeds")
    axis.set_title("Retrieval by crowding")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _load_run_config(input_dir: Path) -> dict[str, object]:
    path = input_dir / "run_config.json"
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("run_config.json must contain a JSON object.")
    return loaded


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.input_dir / "retrieval-diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_path = args.input_dir / "counterfactual_retrieval.csv"
    metadata_path = args.input_dir / "split_metadata.csv"
    if not retrieval_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "Expected counterfactual_retrieval.csv and split_metadata.csv in "
            f"{args.input_dir}."
        )

    retrieval = pd.read_csv(retrieval_path)
    metadata = pd.read_csv(metadata_path)
    run_config = _load_run_config(args.input_dir)
    summary = summarize_retrieval(retrieval)
    family_summary = summarize_retrieval_families(summary)
    merged = merge_test_metadata(retrieval, metadata)

    by_crowding = summarize_retrieval_by(merged, "crowding")
    by_width = summarize_retrieval_by(merged, "stroke_width")
    by_value = summarize_retrieval_by(merged, "stroke_value")
    merged = merged.copy()
    merged["stroke_length_bin"] = pd.cut(
        merged["stroke_length"],
        bins=[0.0, 0.35, 0.55, 0.75, float("inf")],
        labels=["short", "medium", "long", "very_long"],
        include_lowest=True,
    )
    by_length = summarize_retrieval_by(merged, "stroke_length_bin")

    summary.to_csv(output_dir / "retrieval_summary.csv", index=False)
    family_summary.to_csv(
        output_dir / "retrieval_family_summary.csv",
        index=False,
    )
    by_crowding.to_csv(output_dir / "retrieval_by_crowding.csv", index=False)
    by_width.to_csv(output_dir / "retrieval_by_stroke_width.csv", index=False)
    by_value.to_csv(output_dir / "retrieval_by_stroke_value.csv", index=False)
    by_length.to_csv(output_dir / "retrieval_by_stroke_length.csv", index=False)

    _plot_candidate_preferences(
        family_summary,
        output_dir / "candidate_selection_distribution.png",
    )
    _plot_pairwise_wins(
        family_summary,
        output_dir / "pairwise_true_win_rates.png",
    )
    _plot_margins(retrieval, output_dir / "true_margin_distribution.png")
    _plot_by_crowding(by_crowding, output_dir / "retrieval_by_crowding.png")

    formal_requested = bool(run_config.get("formal_run_requested", False))
    formal_eligible = bool(run_config.get("formal_eligible", False))
    analysis_scope = "formal" if formal_requested else "development"
    selected_family_value = run_config.get("selected_model_family_by_validation")
    selected_family = (
        str(selected_family_value) if selected_family_value is not None else None
    )
    if selected_family is not None and selected_family not in set(
        family_summary["model"]
    ):
        raise ValueError(
            "The validation-selected family in run_config.json is absent from "
            "counterfactual_retrieval.csv."
        )

    diagnostic: dict[str, object] = {
        "analysis_scope": analysis_scope,
        "formal_run_requested": formal_requested,
        "formal_eligible": formal_eligible,
        "recorded_gate_status": run_config.get("gate_status"),
        "selected_model_family_by_validation": selected_family,
        "chance_rate": 0.25,
        "frozen_gate_threshold": 0.50,
        "decision_note": (
            "Post-hoc diagnostics cannot change the recorded formal decision."
            if formal_requested
            else "Development diagnostics cannot declare a gate decision."
        ),
    }
    if selected_family is not None:
        selected_family_row = family_summary.loc[
            family_summary["model"] == selected_family
        ].iloc[0]
        selected_seed_rows = summary.loc[
            summary["model"] == selected_family
        ].sort_values("seed")
        diagnostic.update(
            {
                "selected_family_seed_count": int(
                    selected_family_row["seed_count"]
                ),
                "selected_family_mean_top1_accuracy": float(
                    selected_family_row["top1_accuracy_mean"]
                ),
                "selected_family_seed_std_top1_accuracy": float(
                    selected_family_row["top1_accuracy_seed_std"]
                ),
                "selected_family_min_top1_accuracy": float(
                    selected_family_row["top1_accuracy_seed_min"]
                ),
                "selected_family_max_top1_accuracy": float(
                    selected_family_row["top1_accuracy_seed_max"]
                ),
                "selected_seed_results": [
                    {
                        "seed": int(row["seed"]),
                        "top1_correct": int(row["top1_correct"]),
                        "examples": int(row["examples"]),
                        "top1_accuracy": float(row["top1_accuracy"]),
                        "top1_wilson_low": float(row["top1_wilson_low"]),
                        "top1_wilson_high": float(row["top1_wilson_high"]),
                    }
                    for _, row in selected_seed_rows.iterrows()
                ],
            }
        )
    (output_dir / "retrieval_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2),
        encoding="utf-8",
    )

    display_columns = [
        "model",
        "seed",
        "top1_correct",
        "examples",
        "top1_accuracy",
        "top1_wilson_low",
        "top1_wilson_high",
        "predicted_true_rate",
        "predicted_shift_position_rate",
        "predicted_change_width_rate",
        "predicted_change_intensity_rate",
        "true_beats_shift_position_rate",
        "true_beats_change_width_rate",
        "true_beats_change_intensity_rate",
    ]
    family_display_columns = [
        "model",
        "seeds",
        "top1_accuracy_mean",
        "top1_accuracy_seed_std",
        "top1_accuracy_seed_min",
        "top1_accuracy_seed_max",
    ]
    print("\nGate 2 retrieval decomposition by seed:\n")
    print(summary[display_columns].to_string(index=False))
    print("\nRetrieval family summary:\n")
    print(family_summary[family_display_columns].to_string(index=False))
    print(f"\nSaved diagnostics to: {output_dir.resolve()}")
    if formal_requested:
        print(
            "This is a post-hoc formal diagnostic. The recorded gate decision "
            f"remains: {run_config.get('gate_status')}."
        )
    else:
        print("This analysis uses development data and cannot declare a gate result.")


if __name__ == "__main__":
    main()
