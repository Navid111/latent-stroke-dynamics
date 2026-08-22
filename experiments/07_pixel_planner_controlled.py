"""Single frozen six-target Stage 3 pixel-planner comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from latent_stroke_dynamics.learned_pixel_planner import (
    PixelCheckpointMetadata,
    load_pixel_checkpoint,
    state_dict_sha256,
)


TARGET_SEEDS = [20260901, 20260902, 20260903, 20260904, 20260905, 20260906]
PLANNER_SEEDS = [20260920, 20260921, 20260922, 20260923, 20260924, 20260925]
METHODS = ["random", "exact", "learned"]
CHECKPOINT_DIGEST = "e32f3612f7a184e4e9b58f95a987551bd25cdb17ff1bf2b6be40fcf5781ea472"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/stage3-controlled-2026-08-22.json"),
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--controlled-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "frozen",
        "analysis_type": "stage3_controlled_pixel_planner",
        "canvas_size": 64,
        "target_strokes": 20,
        "target_seeds": TARGET_SEEDS,
        "planner_seeds": PLANNER_SEEDS,
        "methods": METHODS,
        "steps": 100,
        "candidates_per_step": 128,
        "prediction_batch_size": 32,
        "checkpoint_state_dict_sha256": CHECKPOINT_DIGEST,
        "output_dir": "outputs/stage3-controlled-2026-08-22",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(
                f"Frozen config mismatch for {key}: expected {value!r}, "
                f"observed {config.get(key)!r}."
            )
    proposal = config["proposal"]
    if proposal != {
        "error_guided_fraction": 0.8,
        "min_length": 0.1,
        "max_length": 0.6,
        "width_choices": [1, 2, 3, 4],
        "value_choices": [0, 32, 64, 96, 128],
    }:
        raise RuntimeError("Frozen candidate proposal does not match the protocol.")
    criteria = config["success_criteria"]
    if criteria != {
        "learned_improves_all_targets": True,
        "minimum_mean_final_mse_reduction_vs_random": 0.2,
        "maximum_mean_final_mse_ratio_to_exact": 1.25,
        "implementation_integrity_required": True,
    }:
        raise RuntimeError("Frozen success criteria do not match the protocol.")
    return config


def validate_checkpoint(config: dict[str, Any]) -> tuple[PixelCheckpointMetadata, str]:
    path = Path(config["checkpoint_path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing controlled checkpoint: {path}")
    model, metadata = load_pixel_checkpoint(path, device="cpu")
    digest = state_dict_sha256(model)
    if digest != CHECKPOINT_DIGEST:
        raise RuntimeError("Checkpoint digest does not match the frozen config.")
    expected = {
        "canvas_size": 64,
        "hidden_dim": 64,
        "parameter_count": 833,
        "model_seed": 11,
        "train_seed": 20260824,
        "validation_seed": 20260825,
        "train_samples": 1000,
        "validation_samples": 200,
        "crowding": (0, 5, 15),
        "epochs": 30,
        "patience": 6,
        "batch_size": 16,
        "best_epoch": 29,
    }
    for name, value in expected.items():
        if getattr(metadata, name) != value:
            raise RuntimeError(f"Checkpoint metadata mismatch for {name}.")
    if metadata.test_rows_used_for_training_or_selection:
        raise RuntimeError("Checkpoint reports forbidden test-row use.")
    return metadata, digest


def controlled_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    final = Path(config["output_dir"])
    temporary = final.parent / f".{final.name}.incomplete"
    return final, temporary


def require_absent(final: Path, temporary: Path) -> None:
    if final.exists():
        raise FileExistsError(f"Controlled output already exists: {final}")
    if temporary.exists():
        raise FileExistsError(
            f"Incomplete controlled output exists: {temporary}. "
            "Preserve it and review before any retry."
        )


def run_one_target(
    config: dict[str, Any],
    index: int,
    target_seed: int,
    planner_seed: int,
    target_dir: Path,
) -> None:
    command = [
        sys.executable,
        "experiments/06_pixel_planner_all_methods_smoke.py",
        "--checkpoint",
        config["checkpoint_path"],
        "--canvas-size",
        str(config["canvas_size"]),
        "--target-strokes",
        str(config["target_strokes"]),
        "--target-seed",
        str(target_seed),
        "--planner-seed",
        str(planner_seed),
        "--steps",
        str(config["steps"]),
        "--candidates",
        str(config["candidates_per_step"]),
        "--prediction-batch-size",
        str(config["prediction_batch_size"]),
        "--gif-scale",
        str(config["gif_scale"]),
        "--output-dir",
        str(target_dir),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path = target_dir.parent / f"target_{index:02d}_terminal.log"
    log_path.write_text(
        completed.stdout + "\n--- STDERR ---\n" + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"Controlled target {index} failed; see {log_path}.")


def validate_subrun(
    config: dict[str, Any],
    target_seed: int,
    planner_seed: int,
    target_dir: Path,
) -> None:
    payload = json.loads((target_dir / "run_config.json").read_text(encoding="utf-8"))
    expected = {
        "target_seed": target_seed,
        "planner_seed": planner_seed,
        "steps": config["steps"],
        "candidates_per_step": config["candidates_per_step"],
        "prediction_batch_size": config["prediction_batch_size"],
        "checkpoint_state_dict_sha256": CHECKPOINT_DIGEST,
        "deterministic_learned_replay_passed": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Subrun mismatch for {key} at {target_dir.name}.")


def collect_results(
    config: dict[str, Any],
    targets_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[pd.DataFrame] = []
    diagnostics: list[pd.DataFrame] = []
    progress_rows: list[dict[str, Any]] = []
    for index, (target_seed, planner_seed) in enumerate(
        zip(TARGET_SEEDS, PLANNER_SEEDS, strict=True), start=1
    ):
        target_id = f"target_{index:02d}"
        target_dir = targets_root / target_id
        validate_subrun(config, target_seed, planner_seed, target_dir)
        summary = pd.read_csv(target_dir / "summary.csv")
        summary.insert(0, "target_id", target_id)
        summary.insert(1, "target_seed", target_seed)
        summary.insert(2, "planner_seed", planner_seed)
        summaries.append(summary)
        learned = pd.read_csv(target_dir / "learned_step_diagnostics.csv")
        learned.insert(0, "target_id", target_id)
        learned.insert(1, "target_seed", target_seed)
        learned.insert(2, "planner_seed", planner_seed)
        diagnostics.append(learned)
        initial = summary.set_index("method")["initial_mse"]
        for method in METHODS:
            progress_rows.append(
                {
                    "target_id": target_id,
                    "method": method,
                    "step": 0,
                    "mse": float(initial.loc[method]),
                }
            )
            progress = pd.read_csv(target_dir / method / "progress.csv")
            for row in progress.itertuples(index=False):
                progress_rows.append(
                    {
                        "target_id": target_id,
                        "method": method,
                        "step": int(row.step),
                        "mse": float(row.mse_after),
                    }
                )
    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(diagnostics, ignore_index=True),
        pd.DataFrame(progress_rows),
    )


def validate_metrics(summary: pd.DataFrame, learned: pd.DataFrame, config: dict[str, Any]) -> None:
    if len(summary) != len(TARGET_SEEDS) * len(METHODS):
        raise RuntimeError("Controlled summary row count is incorrect.")
    shared = [
        "initial_mse",
        "final_mse",
        "final_mae",
        "relative_mse_improvement",
        "improved_steps",
        "elapsed_seconds",
    ]
    if not bool(np.isfinite(summary[shared].to_numpy(dtype=float)).all()):
        raise RuntimeError("Controlled shared metrics are non-finite.")
    learned_summary = summary.loc[summary["method"] == "learned"]
    learned_columns = [
        "exact_top1_rate",
        "exact_top5_rate",
        "mean_exact_rank",
        "mean_exact_regret",
        "max_exact_regret",
    ]
    if len(learned_summary) != len(TARGET_SEEDS) or not bool(
        np.isfinite(learned_summary[learned_columns].to_numpy(dtype=float)).all()
    ):
        raise RuntimeError("Controlled learned summaries are invalid.")
    if len(learned) != len(TARGET_SEEDS) * config["steps"]:
        raise RuntimeError("Controlled learned diagnostic row count is incorrect.")
    diagnostic_columns = [
        "mse_before",
        "mse_after",
        "predicted_selected_mse",
        "exact_best_candidate_mse",
        "exact_selected_rank",
        "exact_regret",
    ]
    if not bool(np.isfinite(learned[diagnostic_columns].to_numpy(dtype=float)).all()):
        raise RuntimeError("Controlled learned step diagnostics are non-finite.")


def aggregate_summary(summary: pd.DataFrame, learned: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        subset = summary.loc[summary["method"] == method]
        row: dict[str, Any] = {
            "method": method,
            "targets": len(subset),
            "mean_initial_mse": float(subset["initial_mse"].mean()),
            "mean_final_mse": float(subset["final_mse"].mean()),
            "std_final_mse": float(subset["final_mse"].std(ddof=1)),
            "mean_final_mae": float(subset["final_mae"].mean()),
            "mean_relative_mse_improvement": float(
                subset["relative_mse_improvement"].mean()
            ),
            "mean_improved_steps": float(subset["improved_steps"].mean()),
            "mean_elapsed_seconds": float(subset["elapsed_seconds"].mean()),
            "exact_top1_rate": None,
            "exact_top5_rate": None,
            "mean_exact_rank": None,
            "mean_exact_regret": None,
            "max_exact_regret": None,
        }
        if method == "learned":
            row.update(
                {
                    "exact_top1_rate": float(learned["exact_top1"].mean()),
                    "exact_top5_rate": float(learned["exact_top5"].mean()),
                    "mean_exact_rank": float(learned["exact_selected_rank"].mean()),
                    "mean_exact_regret": float(learned["exact_regret"].mean()),
                    "max_exact_regret": float(learned["exact_regret"].max()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_decision(summary: pd.DataFrame, learned: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    means = summary.groupby("method")["final_mse"].mean()
    random_mean = float(means.loc["random"])
    exact_mean = float(means.loc["exact"])
    learned_mean = float(means.loc["learned"])
    learned_rows = summary.loc[summary["method"] == "learned"]
    improves_all = bool((learned_rows["final_mse"] < learned_rows["initial_mse"]).all())
    reduction_vs_random = 1.0 - learned_mean / max(random_mean, 1e-12)
    ratio_to_exact = learned_mean / max(exact_mean, 1e-12)
    criteria = config["success_criteria"]
    success = bool(
        improves_all
        and reduction_vs_random
        >= criteria["minimum_mean_final_mse_reduction_vs_random"]
        and ratio_to_exact <= criteria["maximum_mean_final_mse_ratio_to_exact"]
    )
    return {
        "controlled_eligible": True,
        "control_status": "success" if success else "fail",
        "implementation_integrity_passed": True,
        "learned_improved_all_targets": improves_all,
        "mean_random_final_mse": random_mean,
        "mean_exact_final_mse": exact_mean,
        "mean_learned_final_mse": learned_mean,
        "learned_mean_final_mse_reduction_vs_random": reduction_vs_random,
        "learned_mean_final_mse_ratio_to_exact": ratio_to_exact,
        "learned_exact_top1_rate": float(learned["exact_top1"].mean()),
        "learned_exact_top5_rate": float(learned["exact_top5"].mean()),
        "learned_mean_exact_rank": float(learned["exact_selected_rank"].mean()),
        "learned_mean_exact_regret": float(learned["exact_regret"].mean()),
        "learned_max_exact_regret": float(learned["exact_regret"].max()),
        "criteria_frozen_before_run": True,
        "formal_paired_control_result_unchanged": True,
    }


def save_progress_plot(progress: pd.DataFrame, path: Path) -> None:
    colors = {"random": "tab:blue", "exact": "tab:orange", "learned": "tab:green"}
    figure, axis = plt.subplots(figsize=(8, 5))
    grouped = progress.groupby(["method", "step"])["mse"].agg(["mean", "std"])
    for method in METHODS:
        values = grouped.loc[method].reset_index()
        std = values["std"].fillna(0.0)
        axis.plot(values["step"], values["mean"], label=method, color=colors[method])
        axis.fill_between(
            values["step"],
            values["mean"] - std,
            values["mean"] + std,
            color=colors[method],
            alpha=0.15,
        )
    axis.set_xlabel("Executed strokes")
    axis.set_ylabel("Mean target pixel MSE")
    axis.set_title("Controlled Stage 3 progress across six targets")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_montage(targets_root: Path, path: Path) -> None:
    figure, axes = plt.subplots(len(TARGET_SEEDS), 4, figsize=(12, 17))
    titles = ["Target", "Random", "Exact", "Learned"]
    for row in range(len(TARGET_SEEDS)):
        target_dir = targets_root / f"target_{row + 1:02d}"
        image_paths = [
            target_dir / "target.png",
            target_dir / "random/final_canvas.png",
            target_dir / "exact/final_canvas.png",
            target_dir / "learned/final_canvas.png",
        ]
        for column, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                axes[row, column].imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
            if row == 0:
                axes[row, column].set_title(titles[column])
            if column == 0:
                axes[row, column].set_ylabel(f"Target {row + 1}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.validate_only and args.controlled_run:
        raise ValueError("Choose either --validate-only or --controlled-run.")
    config = load_config(args.config)
    metadata, digest = validate_checkpoint(config)
    final, temporary = controlled_paths(config)
    require_absent(final, temporary)

    if args.validate_only:
        print("Controlled Stage 3 validation passed.")
        print(f"Config: {args.config}")
        print(f"Checkpoint digest: {digest}")
        print("No controlled data were generated.")
        return
    if not args.controlled_run:
        raise ValueError("Use --validate-only first, then --controlled-run once.")

    temporary.mkdir(parents=True)
    targets_root = temporary / "targets"
    targets_root.mkdir()
    for index, (target_seed, planner_seed) in enumerate(
        zip(TARGET_SEEDS, PLANNER_SEEDS, strict=True), start=1
    ):
        target_dir = targets_root / f"target_{index:02d}"
        print(f"Running controlled target {index}/6...")
        run_one_target(config, index, target_seed, planner_seed, target_dir)
        print(f"  target {index} complete")

    summary, learned, progress = collect_results(config, targets_root)
    validate_metrics(summary, learned, config)
    aggregate = aggregate_summary(summary, learned)
    decision = make_decision(summary, learned, config)

    summary.to_csv(temporary / "per_target_summary.csv", index=False)
    learned.to_csv(temporary / "learned_step_diagnostics.csv", index=False)
    progress.to_csv(temporary / "progress_by_step.csv", index=False)
    aggregate.to_csv(temporary / "aggregate_summary.csv", index=False)
    pd.DataFrame([decision]).to_csv(temporary / "decision.csv", index=False)
    (temporary / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    run_config = {
        "analysis_type": config["analysis_type"],
        "frozen_config": config,
        "controlled_run_requested": True,
        "controlled_eligible": True,
        "checkpoint_metadata": metadata.to_dict(),
        "checkpoint_state_dict_sha256": digest,
        "result_status": decision["control_status"],
        "formal_paired_control_result_unchanged": True,
    }
    (temporary / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    save_progress_plot(progress, temporary / "aggregate_progress.png")
    save_montage(targets_root, temporary / "final_montage.png")

    final.parent.mkdir(parents=True, exist_ok=True)
    temporary.rename(final)
    print("\nControlled Stage 3 aggregate summary\n")
    print(aggregate.to_string(index=False))
    print("\nControlled decision\n")
    print(json.dumps(decision, indent=2))
    print(f"\nSaved controlled outputs to: {final.resolve()}")
    print("Do not rerun or retune this controlled comparison.")


if __name__ == "__main__":
    main()
