"""Inference-only qualitative painter for the completed Phase B0 predictor."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .extension_training import model_state_sha256
from .high_resolution_replay import replay_strokes_high_resolution
from .painting_cli import TargetPolarity, normalize_target_polarity
from .phase_b_joint_embedding import (
    EXPECTED_TRAINABLE_PARAMETERS,
    MultiScaleActionJointEmbeddingModel,
)
from .phase_b_planning import PhaseBPlanningRun, run_phase_b_planner
from .planning import (
    PlanningRun,
    ProposalConfig,
    load_target,
    pixel_mae,
    pixel_mse,
    run_planner,
)


EXPECTED_PREDICTION_ONLY_ARTIFACT_SHA256 = (
    "d13124a1becb7a11a84eb973a5d3acf72780f9813de6ed9422432778406155b4"
)
EXPECTED_PREDICTION_ONLY_STATE_SHA256 = (
    "c1402cb94faadb504487d7be5295ae1c95a5f0b41dd8485768c9e20c5ed9e462"
)
EXPECTED_SOURCE_EXPERIMENT = "phase-b0-colab-native-development-2026-08-24"
EXPECTED_SOURCE_DECISION = "not_eligible"
DEFAULT_STEPS = 100
DEFAULT_CANDIDATES = 128
DEFAULT_SEED = 20261001
RunLike = PhaseBPlanningRun | PlanningRun


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Qualitative Phase B0 inference supports only CPU or CUDA.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def load_prediction_only_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MultiScaleActionJointEmbeddingModel, dict[str, Any]]:
    """Load only the exact completed cloud-native prediction-only checkpoint."""

    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Phase B0 checkpoint not found: {path}")
    artifact_digest = _file_sha256(path)
    if artifact_digest != EXPECTED_PREDICTION_ONLY_ARTIFACT_SHA256:
        raise RuntimeError(
            "Phase B0 prediction-only checkpoint file SHA-256 mismatch: "
            f"expected {EXPECTED_PREDICTION_ONLY_ARTIFACT_SHA256}, "
            f"received {artifact_digest}."
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("Malformed Phase B0 checkpoint payload.")
    expected_exact = {
        "format_version": 1,
        "architecture": "MultiScaleActionJointEmbeddingModel",
        "variant": "joint_prediction_only",
        "seed": 73,
        "best_epoch": 40,
        "training_device": "cuda:0",
        "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
        "state_sha256": EXPECTED_PREDICTION_ONLY_STATE_SHA256,
    }
    for name, expected in expected_exact.items():
        if payload.get(name) != expected:
            raise ValueError(
                f"Phase B0 prediction-only checkpoint field {name!r} changed."
            )
    best_validation = payload.get("best_validation_loss")
    if not isinstance(best_validation, (int, float)) or not np.isclose(
        float(best_validation),
        0.006895331389387138,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("Phase B0 best validation loss changed.")
    for name in ("progress_training_mean", "progress_training_std"):
        value = payload.get(name)
        if not isinstance(value, (int, float)) or not isfinite(float(value)):
            raise ValueError(f"Phase B0 checkpoint field {name!r} is invalid.")
    if float(payload["progress_training_std"]) <= 0.0:
        raise ValueError("Phase B0 progress standard deviation must be positive.")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Phase B0 checkpoint state_dict is malformed.")

    model = MultiScaleActionJointEmbeddingModel()
    model.load_state_dict(state, strict=True)
    state_digest = model_state_sha256(model)
    if state_digest != EXPECTED_PREDICTION_ONLY_STATE_SHA256:
        raise RuntimeError(
            "Loaded Phase B0 prediction-only model state SHA-256 mismatch: "
            f"expected {EXPECTED_PREDICTION_ONLY_STATE_SHA256}, "
            f"received {state_digest}."
        )
    resolved_device = _resolve_device(str(device))
    model = model.to(resolved_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "source_experiment": EXPECTED_SOURCE_EXPERIMENT,
        "source_decision": EXPECTED_SOURCE_DECISION,
        "artifact_sha256": artifact_digest,
        "state_sha256": state_digest,
        "format_version": int(payload["format_version"]),
        "architecture": str(payload["architecture"]),
        "variant": str(payload["variant"]),
        "seed": int(payload["seed"]),
        "best_epoch": int(payload["best_epoch"]),
        "best_validation_loss": float(payload["best_validation_loss"]),
        "training_device": str(payload["training_device"]),
        "inference_device": str(resolved_device),
        "trainable_parameter_count": int(payload["trainable_parameter_count"]),
        "progress_training_mean": float(payload["progress_training_mean"]),
        "progress_training_std": float(payload["progress_training_std"]),
    }
    return model, metadata


def _validate_request(
    target: Path,
    checkpoint: Path,
    output: Path,
    *,
    polarity: str,
    steps: int,
    candidates: int,
    seed: int,
    prediction_batch_size: int,
    high_res_size: int,
    supersample: int,
) -> Path:
    if not target.is_file():
        raise FileNotFoundError(f"Target image not found: {target}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Phase B0 checkpoint not found: {checkpoint}")
    if polarity not in {"auto", "preserve", "invert"}:
        raise ValueError("polarity must be auto, preserve, or invert.")
    if steps < 1 or steps > 300:
        raise ValueError("steps must lie between 1 and 300.")
    if candidates < 2 or candidates > 256:
        raise ValueError("candidates must lie between 2 and 256.")
    if seed < 0 or prediction_batch_size < 1:
        raise ValueError("seed and prediction batch size are invalid.")
    if high_res_size < 64 or high_res_size > 2048:
        raise ValueError("high_res_size must lie between 64 and 2048.")
    if supersample < 1 or supersample > 4:
        raise ValueError("supersample must lie between 1 and 4.")
    if high_res_size * supersample > 4096:
        raise ValueError("high_res_size times supersample must not exceed 4096.")
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {output}")
    if incomplete.exists():
        raise FileExistsError(f"Preserve existing incomplete output: {incomplete}")
    return incomplete


def _trajectory(run: RunLike) -> np.ndarray:
    values = np.asarray(
        [pixel_mse(run.initial_canvas, run.target)]
        + [float(step.mse_after) for step in run.steps],
        dtype=np.float64,
    )
    if values.shape != (len(run.steps) + 1,) or not np.isfinite(values).all():
        raise RuntimeError("Qualitative trajectory contains invalid MSE values.")
    return values


def _summarize_run(run: RunLike, method: str, elapsed: float) -> dict[str, Any]:
    values = _trajectory(run)
    best_step = int(np.argmin(values))
    initial = float(values[0])
    final = float(values[-1])
    row: dict[str, Any] = {
        "method": method,
        "executed_steps": len(run.steps),
        "candidates_per_step": int(run.steps[0].candidate_count),
        "initial_mse": initial,
        "best_step": best_step,
        "best_mse": float(values[best_step]),
        "final_mse": final,
        "final_mae": pixel_mae(run.final_canvas, run.target),
        "relative_final_mse_improvement": (initial - final) / max(initial, 1e-12),
        "improving_steps": int(sum(step.improved for step in run.steps)),
        "elapsed_seconds": float(elapsed),
        "exact_top1_rate": None,
        "exact_top5_rate": None,
        "mean_exact_rank": None,
        "mean_exact_regret": None,
        "mean_score_exact_spearman": None,
    }
    if isinstance(run, PhaseBPlanningRun):
        row.update(
            {
                "exact_top1_rate": float(np.mean([step.exact_top1 for step in run.steps])),
                "exact_top5_rate": float(np.mean([step.exact_top5 for step in run.steps])),
                "mean_exact_rank": float(
                    np.mean([step.exact_selected_rank for step in run.steps])
                ),
                "mean_exact_regret": float(
                    np.mean([step.exact_regret for step in run.steps])
                ),
                "mean_score_exact_spearman": float(
                    np.mean([step.score_exact_spearman for step in run.steps])
                ),
            }
        )
    for name, value in row.items():
        if isinstance(value, float) and not isfinite(value):
            raise RuntimeError(f"Non-finite qualitative summary metric: {name}")
    return row


def _flatten_steps(run: RunLike) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in run.steps:
        row = asdict(step)
        stroke = row.pop("stroke")
        for name, value in stroke.items():
            row[f"stroke_{name}"] = value
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty qualitative CSV.")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_gif(frames: Sequence[Image.Image], path: Path) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=list(frames[1:]),
        duration=140,
        loop=0,
        optimize=False,
    )


def _save_method_artifacts(
    run: RunLike,
    directory: Path,
    *,
    high_res_size: int,
    supersample: int,
) -> None:
    directory.mkdir(parents=False, exist_ok=False)
    values = _trajectory(run)
    best_step = int(np.argmin(values))
    if len(run.frames) != len(run.steps) + 1:
        raise RuntimeError("Qualitative output requires every low-resolution frame.")
    run.initial_canvas.save(directory / "initial_64.png")
    run.frames[best_step].save(directory / "best_64.png")
    run.final_canvas.save(directory / "final_64.png")

    frame_dir = directory / "frames_64"
    frame_dir.mkdir()
    for index, frame in enumerate(run.frames):
        frame.save(frame_dir / f"frame_{index:04d}.png")

    rows = _flatten_steps(run)
    _write_csv(directory / "steps.csv", rows)
    strokes = [
        {
            "step": int(row["step"]),
            "selected_index": int(row["selected_index"]),
            "x0": float(row["stroke_x0"]),
            "y0": float(row["stroke_y0"]),
            "x1": float(row["stroke_x1"]),
            "y1": float(row["stroke_y1"]),
            "width": int(row["stroke_width"]),
            "value": int(row["stroke_value"]),
        }
        for row in rows
    ]
    (directory / "strokes.json").write_text(
        json.dumps(strokes, indent=2),
        encoding="utf-8",
    )

    high_res = replay_strokes_high_resolution(
        tuple(step.stroke for step in run.steps),
        output_size=high_res_size,
        planning_size=64,
        supersample=supersample,
    )
    high_res[0].save(directory / f"initial_{high_res_size}.png")
    high_res[best_step].save(directory / f"best_{high_res_size}.png")
    high_res[-1].save(directory / f"final_{high_res_size}.png")
    _save_gif(high_res, directory / f"painting_{high_res_size}.gif")


def _save_progress_plot(
    latent: PhaseBPlanningRun,
    exact: PlanningRun,
    output: Path,
) -> None:
    latent_values = _trajectory(latent)
    exact_values = _trajectory(exact)
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(latent_values, label="Phase B0 prediction-only latent", linewidth=2)
    axis.plot(exact_values, label="Exact pixel greedy", linewidth=2)
    axis.scatter([int(np.argmin(latent_values))], [float(latent_values.min())], color="black", s=25)
    axis.set_xlabel("Executed strokes")
    axis.set_ylabel("Target pixel MSE (diagnostic only)")
    axis.set_title("Qualitative latent-planner comparison")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_comparison_plot(
    target: Image.Image,
    latent: PhaseBPlanningRun,
    exact: PlanningRun,
    output: Path,
) -> None:
    latent_values = _trajectory(latent)
    exact_values = _trajectory(exact)
    latent_best = int(np.argmin(latent_values))
    exact_best = int(np.argmin(exact_values))
    images = (
        (target, "Target"),
        (latent.frames[latent_best], f"Latent best: {latent_best}\nMSE {latent_values[latent_best]:.5f}"),
        (latent.final_canvas, f"Latent final\nMSE {latent_values[-1]:.5f}"),
        (exact.frames[exact_best], f"Exact best: {exact_best}\nMSE {exact_values[exact_best]:.5f}"),
        (exact.final_canvas, f"Exact final\nMSE {exact_values[-1]:.5f}"),
    )
    figure, axes = plt.subplots(1, len(images), figsize=(17, 3.7))
    for axis, (image, title) in zip(axes, images, strict=True):
        axis.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_prediction_only_qualitative_comparison(
    target_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    polarity: TargetPolarity = "auto",
    steps: int = DEFAULT_STEPS,
    candidates: int = DEFAULT_CANDIDATES,
    seed: int = DEFAULT_SEED,
    prediction_batch_size: int = 32,
    device: str = "auto",
    high_res_size: int = 512,
    supersample: int = 2,
) -> tuple[Path, dict[str, Any]]:
    """Compare the frozen prediction-only latent model with exact pixel greedy."""

    target_path = Path(target_path).expanduser()
    checkpoint_path = Path(checkpoint_path).expanduser()
    output_dir = Path(output_dir).expanduser()
    incomplete = _validate_request(
        target_path,
        checkpoint_path,
        output_dir,
        polarity=polarity,
        steps=steps,
        candidates=candidates,
        seed=seed,
        prediction_batch_size=prediction_batch_size,
        high_res_size=high_res_size,
        supersample=supersample,
    )
    resolved_device = _resolve_device(device)
    model, checkpoint_metadata = load_prediction_only_checkpoint(
        checkpoint_path,
        device=resolved_device,
    )
    checkpoint_hash_before = _file_sha256(checkpoint_path)
    target_hash_before = _file_sha256(target_path)

    before_polarity = load_target(target_path, size=64)
    target, inverted, border_median = normalize_target_polarity(
        before_polarity,
        polarity,
    )
    initial_mse = pixel_mse(Image.new("L", (64, 64), 255), target)
    if initial_mse <= 1e-12:
        raise ValueError("The normalized target is blank white; no painting is needed.")

    proposal = ProposalConfig(count=candidates)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    incomplete.mkdir(parents=False, exist_ok=False)

    started = time.perf_counter()
    latent = run_phase_b_planner(
        target,
        model,
        mode="prediction",
        maximum_steps=steps,
        seed=seed,
        proposal_config=proposal,
        prediction_batch_size=prediction_batch_size,
        allow_no_op=False,
        capture_frames=True,
    )
    latent_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    exact = run_planner(
        target,
        "exact",
        steps=steps,
        seed=seed,
        proposal_config=proposal,
        capture_frames=True,
    )
    exact_elapsed = time.perf_counter() - started

    if _file_sha256(checkpoint_path) != checkpoint_hash_before:
        raise RuntimeError("The Phase B0 checkpoint changed during inference.")
    if _file_sha256(target_path) != target_hash_before:
        raise RuntimeError("The input target changed during inference.")

    latent_summary = _summarize_run(
        latent,
        "phase_b0_joint_prediction_only_forced_qualitative",
        latent_elapsed,
    )
    exact_summary = _summarize_run(exact, "exact_pixel_qualitative", exact_elapsed)
    summary: dict[str, Any] = {
        "status": "phase_b0_prediction_only_qualitative_inference_complete",
        "evidential_role": "exploratory_qualitative_inference_only",
        "source_phase_b0_decision": EXPECTED_SOURCE_DECISION,
        "latent": latent_summary,
        "exact_pixel": exact_summary,
        "latent_final_mse_ratio_to_exact": latent_summary["final_mse"]
        / max(float(exact_summary["final_mse"]), 1e-12),
        "latent_best_mse_ratio_to_exact": latent_summary["best_mse"]
        / max(float(exact_summary["best_mse"]), 1e-12),
        "target_polarity_inverted": inverted,
        "target_border_median_before_polarity": border_median,
        "checkpoint_unchanged": True,
        "target_file_unchanged": True,
        "models_trained": False,
        "formal_claims_allowed": False,
        "historical_decision_changed": False,
    }

    before_polarity.save(incomplete / "target_before_polarity_64.png")
    target.save(incomplete / "target_64.png")
    target.resize(
        (high_res_size, high_res_size),
        resample=Image.Resampling.LANCZOS,
    ).save(incomplete / f"target_{high_res_size}.png")
    _save_method_artifacts(
        latent,
        incomplete / "latent_prediction_only",
        high_res_size=high_res_size,
        supersample=supersample,
    )
    _save_method_artifacts(
        exact,
        incomplete / "exact_pixel",
        high_res_size=high_res_size,
        supersample=supersample,
    )
    _save_progress_plot(latent, exact, incomplete / "progress_comparison.png")
    _save_comparison_plot(target, latent, exact, incomplete / "comparison.png")
    (incomplete / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_csv(
        incomplete / "summary.csv",
        (latent_summary, exact_summary),
    )
    run_config = {
        "status": summary["status"],
        "qualitative_inference_only": True,
        "source_experiment": EXPECTED_SOURCE_EXPERIMENT,
        "source_decision_preserved": EXPECTED_SOURCE_DECISION,
        "formal_phase_b0_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
        "rerun_of_completed_training": False,
        "training_performed": False,
        "model_selection_performed": False,
        "checkpoint": checkpoint_metadata,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "input_target": str(target_path.resolve()),
        "input_target_sha256": target_hash_before,
        "target_processing": {
            "center_crop_square": True,
            "grayscale": True,
            "planning_size": 64,
            "polarity_requested": polarity,
            "polarity_inverted": inverted,
            "border_median_before_polarity": border_median,
        },
        "planner": {
            "mode": "prediction",
            "forced_steps": steps,
            "candidates_per_step": candidates,
            "seed": seed,
            "prediction_batch_size": prediction_batch_size,
            "proposal": asdict(proposal),
            "exact_renderer_execution": True,
        },
        "presentation_replay": {
            "output_size": high_res_size,
            "supersample": supersample,
            "stroke_sequence_changed": False,
        },
        "inference_device": str(resolved_device),
    }
    (incomplete / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    incomplete.replace(output_dir)
    return output_dir, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--polarity", choices=("auto", "preserve", "invert"), default="auto")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prediction-batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--high-res-size", type=int, default=512)
    parser.add_argument("--supersample", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output, summary = run_prediction_only_qualitative_comparison(
        args.target,
        args.checkpoint,
        args.output_dir,
        polarity=args.polarity,
        steps=args.steps,
        candidates=args.candidates,
        seed=args.seed,
        prediction_batch_size=args.prediction_batch_size,
        device=args.device,
        high_res_size=args.high_res_size,
        supersample=args.supersample,
    )
    print("\nPhase B0 prediction-only qualitative inference complete\n")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved artifacts to: {output.resolve()}")
    print("No training or formal evaluation was performed; the not-eligible decision is unchanged.")


if __name__ == "__main__":
    main()
