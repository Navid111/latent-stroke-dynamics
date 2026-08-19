"""Gate 1: test whether frozen visual features notice controlled stroke changes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from latent_stroke_dynamics.encoder import FrozenVisionEncoder
from latent_stroke_dynamics.gate1 import (
    COMPARISON_ORDER,
    STRUCTURAL_COMPARISONS,
    ComparisonPair,
    build_pairs,
    changed_pixel_count,
    patch_change_mask,
    patch_summary_metrics,
    pixel_distance,
    stroke_length_normalized,
)
from latent_stroke_dynamics.metrics import alternating_pair_distances


AGGREGATE_METRICS = [
    "pixel_mean_absolute_difference",
    "global_cosine_distance",
    "patch_mean_cosine_distance",
    "patch_max_cosine_distance",
    "patch_top10pct_mean_cosine_distance",
    "patch_changed_region_mean_cosine_distance",
    "patch_unchanged_region_mean_cosine_distance",
    "changed_patch_fraction",
    "localization_enrichment",
    "localization_topk_recall",
    "localization_topk_lift",
    "patch_reference_region_mean_cosine_distance",
    "patch_reference_outside_mean_cosine_distance",
    "reference_region_enrichment",
]


def _set_boxplot_labels(axis: plt.Axes, labels: list[str]) -> None:
    axis.set_xticks(range(1, len(labels) + 1))
    axis.set_xticklabels(labels)
    axis.tick_params(axis="x", rotation=42, labelsize=8)
    axis.grid(axis="y", alpha=0.25)


def save_distribution_plot(results: pd.DataFrame, output_path: Path) -> None:
    """Save distance plots with crowding levels separated into rows."""

    metrics = [
        ("pixel_mean_absolute_difference", "Pixel mean absolute difference"),
        ("global_cosine_distance", "Global-token cosine distance"),
        ("patch_mean_cosine_distance", "Mean patch-token cosine distance"),
        ("patch_top10pct_mean_cosine_distance", "Top-10% patch distance"),
        (
            "patch_reference_region_mean_cosine_distance",
            "Reference stroke-region distance",
        ),
    ]
    crowding_levels = sorted(int(value) for value in results["crowding"].unique())
    labels = [name for name in COMPARISON_ORDER if name in set(results["comparison"])]

    figure, axes = plt.subplots(
        len(crowding_levels),
        len(metrics),
        figsize=(5.2 * len(metrics), 4.5 * len(crowding_levels)),
        squeeze=False,
    )

    for row, crowding in enumerate(crowding_levels):
        subset = results.loc[results["crowding"] == crowding]
        for column, (metric, title) in enumerate(metrics):
            axis = axes[row, column]
            data = [subset.loc[subset["comparison"] == name, metric].values for name in labels]
            axis.boxplot(data, showfliers=False)
            _set_boxplot_labels(axis, labels)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"Prior strokes = {crowding}")

    figure.suptitle("Gate 1 distances separated by canvas crowding", y=1.005)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_localization_plot(results: pd.DataFrame, output_path: Path) -> None:
    """Save quantitative spatial-localization diagnostics."""

    metrics = [
        ("patch_changed_region_mean_cosine_distance", "Changed-region distance"),
        ("localization_topk_recall", "Top-k localization recall"),
        ("localization_topk_lift", "Top-k lift over random"),
    ]
    crowding_levels = sorted(int(value) for value in results["crowding"].unique())
    labels = [name for name in STRUCTURAL_COMPARISONS if name in set(results["comparison"])]

    figure, axes = plt.subplots(
        len(crowding_levels),
        len(metrics),
        figsize=(5.2 * len(metrics), 4.5 * len(crowding_levels)),
        squeeze=False,
    )

    for row, crowding in enumerate(crowding_levels):
        subset = results.loc[results["crowding"] == crowding]
        for column, (metric, title) in enumerate(metrics):
            axis = axes[row, column]
            data = [
                subset.loc[subset["comparison"] == name, metric]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .values
                for name in labels
            ]
            axis.boxplot(data, showfliers=False)
            _set_boxplot_labels(axis, labels)
            if row == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(f"Prior strokes = {crowding}")

    figure.suptitle("Where did the spatial representation change?", y=1.005)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_example_heatmap(
    pair: ComparisonPair,
    patch_distances: torch.Tensor,
    patch_grid: tuple[int, int],
    output_path: Path,
) -> None:
    heatmap = patch_distances.reshape(*patch_grid).numpy()
    changed_patches = patch_change_mask(pair.before, pair.after, patch_grid).reshape(
        *patch_grid
    )
    before = np.asarray(pair.before)
    after = np.asarray(pair.after)
    difference = np.abs(after.astype(np.float32) - before.astype(np.float32))

    figure, axes = plt.subplots(1, 5, figsize=(17, 3.5))
    axes[0].imshow(before, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Before")
    axes[1].imshow(after, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("After")
    axes[2].imshow(difference, cmap="magma")
    axes[2].set_title("Pixel change")
    axes[3].imshow(changed_patches, cmap="gray", vmin=0, vmax=1)
    axes[3].set_title("Changed patch mask")
    image = axes[4].imshow(heatmap, cmap="magma", interpolation="nearest")
    axes[4].set_title("Patch-feature change")
    figure.colorbar(image, ax=axes[4], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{pair.comparison}; prior strokes = {pair.crowding}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_gate_diagnostics(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize the paired pre-run engineering criteria by crowding level."""

    rows: list[dict[str, float | int]] = []
    for crowding in sorted(int(value) for value in results["crowding"].unique()):
        subset = results.loc[results["crowding"] == crowding]

        def paired_win_rate(metric: str) -> float:
            pivot = subset.pivot(index="sample_id", columns="comparison", values=metric)
            valid = pivot[["add_stroke", "sparse_pixel_matched_noise"]].dropna()
            return float(
                (valid["add_stroke"] > valid["sparse_pixel_matched_noise"]).mean()
            )

        add_rows = subset.loc[subset["comparison"] == "add_stroke"]
        no_change_rows = subset.loc[subset["comparison"] == "no_change"]
        rows.append(
            {
                "crowding": crowding,
                "samples": int(add_rows["sample_id"].nunique()),
                "add_vs_sparse_top10_win_rate": paired_win_rate(
                    "patch_top10pct_mean_cosine_distance"
                ),
                "add_vs_sparse_reference_region_win_rate": paired_win_rate(
                    "patch_reference_region_mean_cosine_distance"
                ),
                "add_median_localization_topk_lift": float(
                    add_rows["localization_topk_lift"].median()
                ),
                "add_median_reference_region_enrichment": float(
                    add_rows["reference_region_enrichment"].median()
                ),
                "no_change_max_patch_distance": float(
                    no_change_rows["patch_max_cosine_distance"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--samples", type=int, default=10, help="Paired samples per crowding level")
    parser.add_argument("--canvas-size", type=int, default=64)
    parser.add_argument("--crowding", nargs="+", type=int, default=[0, 5, 15])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gate1"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be positive.")
    if any(level < 0 for level in args.crowding):
        raise ValueError("Crowding levels cannot be negative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pairs = build_pairs(args.samples, args.canvas_size, args.crowding, args.seed)
    flattened_images = [image for pair in pairs for image in (pair.before, pair.after)]

    encoder = FrozenVisionEncoder(model_name=args.model, device=args.device)
    encodings = encoder.encode(flattened_images, batch_size=args.batch_size)

    global_distances = alternating_pair_distances(encodings.global_features)
    patch_maps = alternating_pair_distances(encodings.patch_features)

    rows: list[dict[str, float | int | str]] = []
    for index, pair in enumerate(pairs):
        patch_metrics = patch_summary_metrics(
            patch_maps[index],
            pair.before,
            pair.after,
            encodings.patch_grid,
            reference_mask=pair.reference_mask,
        )
        pixels_changed = changed_pixel_count(pair.before, pair.after)
        rows.append(
            {
                "comparison": pair.comparison,
                "crowding": pair.crowding,
                "sample_id": pair.sample_id,
                "stroke_x0": pair.stroke.x0,
                "stroke_y0": pair.stroke.y0,
                "stroke_x1": pair.stroke.x1,
                "stroke_y1": pair.stroke.y1,
                "stroke_width": pair.stroke.width,
                "stroke_value": pair.stroke.value,
                "stroke_length_normalized": stroke_length_normalized(pair.stroke),
                "changed_pixel_count": pixels_changed,
                "changed_pixel_fraction": pixels_changed / (pair.before.width * pair.before.height),
                "pixel_mean_absolute_difference": pixel_distance(pair.before, pair.after),
                "global_cosine_distance": float(global_distances[index]),
                **patch_metrics,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "results.csv", index=False)

    aggregate = (
        results.groupby(["comparison", "crowding"], sort=False)[AGGREGATE_METRICS]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.to_csv(args.output_dir / "aggregate_summary.csv", index=False)

    gate_diagnostics = build_gate_diagnostics(results)
    gate_diagnostics.to_csv(args.output_dir / "gate_diagnostics.csv", index=False)

    save_distribution_plot(results, args.output_dir / "distance_distributions.png")
    save_localization_plot(results, args.output_dir / "localization_metrics.png")

    for crowding in sorted(set(args.crowding)):
        example_index = next(
            index
            for index, pair in enumerate(pairs)
            if pair.comparison == "add_stroke"
            and pair.crowding == crowding
            and pair.sample_id == 0
        )
        save_example_heatmap(
            pairs[example_index],
            patch_maps[example_index],
            encodings.patch_grid,
            args.output_dir / f"example_patch_heatmap_crowding_{crowding}.png",
        )

    config = {
        "model": args.model,
        "samples_per_crowding_level": args.samples,
        "canvas_size": args.canvas_size,
        "crowding_levels": sorted(set(args.crowding)),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(encoder.device),
        "patch_grid": list(encodings.patch_grid),
        "comparison_pairs": len(pairs),
        "comparisons": list(COMPARISON_ORDER),
        "paired_across_crowding": True,
        "canonical_test_stroke": {"width": 2, "value": 0, "min_length": 0.35},
        "gate_control": "sparse_pixel_matched_noise",
        "dense_pixel_matched_noise_role": "stress_test",
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\nGate diagnostics by crowding:\n")
    print(gate_diagnostics.to_string(index=False))
    print("\nMean diagnostics by condition and crowding:\n")
    display_columns = [
        "global_cosine_distance",
        "patch_top10pct_mean_cosine_distance",
        "patch_reference_region_mean_cosine_distance",
        "localization_topk_lift",
    ]
    print(
        results.groupby(["comparison", "crowding"])[display_columns]
        .mean()
        .to_string()
    )
    print(f"\nSaved Gate 1 results to: {args.output_dir.resolve()}")
    print(
        "This is a representation diagnostic, not an automatic pass/fail decision. "
        "Inspect paired separation, localization, and crowding robustness together."
    )


if __name__ == "__main__":
    main()
