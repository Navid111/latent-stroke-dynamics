"""Gate 1: test whether frozen visual features notice controlled stroke changes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from latent_stroke_dynamics.encoder import FrozenVisionEncoder
from latent_stroke_dynamics.metrics import alternating_pair_distances
from latent_stroke_dynamics.renderer import (
    Stroke,
    random_base_canvas,
    render_stroke,
    sample_stroke,
)


COMPARISON_ORDER = [
    "no_change",
    "tiny_pixel_noise",
    "add_stroke",
    "shift_position",
    "change_width",
    "change_intensity",
]


@dataclass(frozen=True)
class ComparisonPair:
    before: Image.Image
    after: Image.Image
    comparison: str
    crowding: int
    sample_id: int


def tiny_noise(image: Image.Image, rng: np.random.Generator, sigma: float = 1.25) -> Image.Image:
    values = np.asarray(image, dtype=np.float32)
    noisy = np.clip(values + rng.normal(0.0, sigma, size=values.shape), 0, 255)
    return Image.fromarray(noisy.astype(np.uint8), mode="L")


def build_pairs(
    samples: int,
    canvas_size: int,
    crowding_levels: list[int],
    seed: int,
) -> list[ComparisonPair]:
    rng = np.random.default_rng(seed)
    pairs: list[ComparisonPair] = []

    for crowding in crowding_levels:
        for sample_id in range(samples):
            base = random_base_canvas(canvas_size, crowding, rng)
            stroke = sample_stroke(rng)

            added = render_stroke(base, stroke)
            shifted = render_stroke(base, stroke.shifted(dx=0.08, dy=-0.06))
            thin = render_stroke(base, replace(stroke, width=1))
            thick = render_stroke(base, replace(stroke, width=max(5, stroke.width + 3)))
            dark = render_stroke(base, replace(stroke, value=16))
            light = render_stroke(base, replace(stroke, value=176))

            pairs.extend(
                [
                    ComparisonPair(base, base.copy(), "no_change", crowding, sample_id),
                    ComparisonPair(
                        base,
                        tiny_noise(base, rng),
                        "tiny_pixel_noise",
                        crowding,
                        sample_id,
                    ),
                    ComparisonPair(base, added, "add_stroke", crowding, sample_id),
                    ComparisonPair(added, shifted, "shift_position", crowding, sample_id),
                    ComparisonPair(thin, thick, "change_width", crowding, sample_id),
                    ComparisonPair(dark, light, "change_intensity", crowding, sample_id),
                ]
            )

    return pairs


def pixel_distance(before: Image.Image, after: Image.Image) -> float:
    left = np.asarray(before, dtype=np.float32)
    right = np.asarray(after, dtype=np.float32)
    return float(np.mean(np.abs(left - right)) / 255.0)


def save_distribution_plot(results: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("pixel_mean_absolute_difference", "Pixel mean absolute difference"),
        ("global_cosine_distance", "Global-token cosine distance"),
        ("patch_mean_cosine_distance", "Mean patch-token cosine distance"),
    ]
    available = [name for name in COMPARISON_ORDER if name in set(results["comparison"])]

    figure, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    for axis, (metric, title) in zip(axes, metrics):
        data = [results.loc[results["comparison"] == name, metric].values for name in available]
        axis.boxplot(data, labels=available, showfliers=False)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=40)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_example_heatmap(
    pair: ComparisonPair,
    patch_distances: torch.Tensor,
    patch_grid: tuple[int, int],
    output_path: Path,
) -> None:
    heatmap = patch_distances.reshape(*patch_grid).numpy()
    before = np.asarray(pair.before)
    after = np.asarray(pair.after)
    difference = np.abs(after.astype(np.float32) - before.astype(np.float32))

    figure, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    axes[0].imshow(before, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Before")
    axes[1].imshow(after, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("After")
    axes[2].imshow(difference, cmap="magma")
    axes[2].set_title("Pixel change")
    image = axes[3].imshow(heatmap, cmap="magma", interpolation="nearest")
    axes[3].set_title("Patch-feature change")
    figure.colorbar(image, ax=axes[3], fraction=0.046, pad=0.04)
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"{pair.comparison}; prior strokes = {pair.crowding}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--samples", type=int, default=10, help="Samples per crowding level")
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
    patch_mean_distances = patch_maps.mean(dim=-1)
    patch_max_distances = patch_maps.max(dim=-1).values

    rows: list[dict[str, float | int | str]] = []
    for index, pair in enumerate(pairs):
        rows.append(
            {
                "comparison": pair.comparison,
                "crowding": pair.crowding,
                "sample_id": pair.sample_id,
                "pixel_mean_absolute_difference": pixel_distance(pair.before, pair.after),
                "global_cosine_distance": float(global_distances[index]),
                "patch_mean_cosine_distance": float(patch_mean_distances[index]),
                "patch_max_cosine_distance": float(patch_max_distances[index]),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(args.output_dir / "results.csv", index=False)

    aggregate = (
        results.groupby(["comparison", "crowding"], sort=False)[
            [
                "pixel_mean_absolute_difference",
                "global_cosine_distance",
                "patch_mean_cosine_distance",
                "patch_max_cosine_distance",
            ]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.to_csv(args.output_dir / "aggregate_summary.csv", index=False)

    save_distribution_plot(results, args.output_dir / "distance_distributions.png")
    example_index = next(
        index
        for index, pair in enumerate(pairs)
        if pair.comparison == "add_stroke" and pair.crowding == min(args.crowding)
    )
    save_example_heatmap(
        pairs[example_index],
        patch_maps[example_index],
        encodings.patch_grid,
        args.output_dir / "example_patch_heatmap.png",
    )

    config = {
        "model": args.model,
        "samples_per_crowding_level": args.samples,
        "canvas_size": args.canvas_size,
        "crowding_levels": args.crowding,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "device": str(encoder.device),
        "patch_grid": list(encodings.patch_grid),
        "comparison_pairs": len(pairs),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    display_columns = [
        "comparison",
        "crowding",
        "global_cosine_distance",
        "patch_mean_cosine_distance",
    ]
    print("\nMean distances by condition:\n")
    print(results[display_columns].groupby(["comparison", "crowding"]).mean().to_string())
    print(f"\nSaved Gate 1 results to: {args.output_dir.resolve()}")
    print(
        "Inspect the distributions and heatmap before training a predictor; "
        "a non-zero number alone is not evidence that the representation is useful."
    )


if __name__ == "__main__":
    main()
