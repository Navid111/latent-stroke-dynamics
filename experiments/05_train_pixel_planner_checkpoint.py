"""Train and save the separate Stage 3 pixel-planner demonstration checkpoint."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from latent_stroke_dynamics.gate2 import (
    PRIMARY_CROWDING,
    build_transition_split,
    parameter_count,
    transition_fingerprint,
)
from latent_stroke_dynamics.learned_pixel_planner import (
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_TYPE,
    PixelCheckpointMetadata,
    load_pixel_checkpoint,
    save_pixel_checkpoint,
    state_dict_sha256,
)
from latent_stroke_dynamics.pixel_control import (
    PIXEL_INPUT_DIM,
    MLPPixelDeltaPredictor,
    PixelTensors,
    balanced_pixel_mse,
    build_pixel_tensors,
)


CANVAS_SIZE = 64
TRAIN_SAMPLES = 1000
VALIDATION_SAMPLES = 200
TRAIN_SEED = 20260824
VALIDATION_SEED = 20260825
MODEL_SEED = 11
HIDDEN_DIM = 64
EPOCHS = 30
PATIENCE = 6
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/stage3-pixel-mlp-seed11.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stage3-demo-checkpoint"),
    )
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def batch_tensors(
    tensors: PixelTensors,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current = tensors.current[indices].float()
    next_canvas = tensors.next_canvas[indices].float()
    actions = tensors.actions[indices].float()
    masks = tensors.action_masks[indices].float()
    return current, next_canvas, actions, masks


def mean_loss(
    model: MLPPixelDeltaPredictor,
    tensors: PixelTensors,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(tensors.current), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(tensors.current))
            indices = torch.arange(start, stop)
            current, next_canvas, actions, masks = batch_tensors(tensors, indices)
            per_example = balanced_pixel_mse(
                model(current, actions, masks),
                next_canvas - current,
                masks,
                reduction="none",
            )
            total += float(per_example.sum())
            count += len(indices)
    return total / count


def main() -> None:
    args = parse_args()
    if args.threads < 0:
        raise ValueError("--threads cannot be negative.")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("Generating train and validation transitions only...")
    train_examples = build_transition_split(
        TRAIN_SAMPLES,
        CANVAS_SIZE,
        PRIMARY_CROWDING,
        TRAIN_SEED,
    )
    validation_examples = build_transition_split(
        VALIDATION_SAMPLES,
        CANVAS_SIZE,
        PRIMARY_CROWDING,
        VALIDATION_SEED,
    )
    train_fingerprints = {transition_fingerprint(item) for item in train_examples}
    validation_fingerprints = {
        transition_fingerprint(item) for item in validation_examples
    }
    if train_fingerprints.intersection(validation_fingerprints):
        raise RuntimeError("Train and validation transitions overlap.")
    train_tensors = build_pixel_tensors(train_examples, CANVAS_SIZE)
    validation_tensors = build_pixel_tensors(validation_examples, CANVAS_SIZE)

    random.seed(MODEL_SEED)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    model = MLPPixelDeltaPredictor(HIDDEN_DIM)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator().manual_seed(MODEL_SEED)
    best_validation = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, int | float]] = []

    print("Training demonstration MLP on CPU...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(TRAIN_SAMPLES, generator=generator)
        train_total = 0.0
        train_count = 0
        for start in range(0, TRAIN_SAMPLES, BATCH_SIZE):
            indices = permutation[start : start + BATCH_SIZE]
            current, next_canvas, actions, masks = batch_tensors(train_tensors, indices)
            loss = balanced_pixel_mse(
                model(current, actions, masks),
                next_canvas - current,
                masks,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_total += float(loss.detach()) * len(indices)
            train_count += len(indices)

        train_loss = train_total / train_count
        validation_loss = mean_loss(model, validation_tensors)
        history.append(
            {
                "epoch": epoch,
                "train_balanced_mse": train_loss,
                "validation_balanced_mse": validation_loss,
            }
        )
        print(
            f"  epoch {epoch:02d}: train={train_loss:.8f}, "
            f"validation={validation_loss:.8f}"
        )
        if validation_loss < best_validation - 1e-12:
            best_validation = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                print(f"  early-stopped at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    model.eval()
    selected_validation = mean_loss(model, validation_tensors)
    if not np.isclose(selected_validation, best_validation, rtol=0.0, atol=1e-12):
        raise RuntimeError("Selected state does not reproduce best validation loss.")

    metadata = PixelCheckpointMetadata(
        checkpoint_type=CHECKPOINT_TYPE,
        format_version=CHECKPOINT_FORMAT_VERSION,
        canvas_size=CANVAS_SIZE,
        pixel_input_dim=PIXEL_INPUT_DIM,
        action_dim=7,
        architecture="MLPPixelDeltaPredictor",
        hidden_dim=HIDDEN_DIM,
        parameter_count=parameter_count(model),
        model_seed=MODEL_SEED,
        train_seed=TRAIN_SEED,
        validation_seed=VALIDATION_SEED,
        train_samples=TRAIN_SAMPLES,
        validation_samples=VALIDATION_SAMPLES,
        crowding=PRIMARY_CROWDING,
        epochs=EPOCHS,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        batch_size=BATCH_SIZE,
        best_epoch=best_epoch,
        best_validation_loss=best_validation,
        test_rows_used_for_training_or_selection=False,
    )
    checkpoint_path = save_pixel_checkpoint(model, metadata, args.checkpoint)
    loaded_model, loaded_metadata = load_pixel_checkpoint(checkpoint_path)

    preview_indices = torch.arange(4)
    current, _, actions, masks = batch_tensors(validation_tensors, preview_indices)
    with torch.inference_mode():
        original_prediction = model(current, actions, masks)
        loaded_prediction = loaded_model(current, actions, masks)
    reload_predictions_identical = bool(
        torch.equal(original_prediction, loaded_prediction)
    )
    if loaded_metadata != metadata or not reload_predictions_identical:
        raise RuntimeError("Saved checkpoint did not reload identically.")

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(args.output_dir / "training_history.csv", index=False)
    (args.output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata.to_dict(), indent=2),
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - started
    summary = {
        "status": "success",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_type": CHECKPOINT_TYPE,
        "state_dict_sha256": state_dict_sha256(loaded_model),
        "parameter_count": parameter_count(loaded_model),
        "best_epoch": best_epoch,
        "best_validation_balanced_mse": best_validation,
        "epochs_completed": len(history),
        "train_validation_overlap_count": 0,
        "test_rows_generated": 0,
        "test_rows_used_for_training_or_selection": False,
        "reload_predictions_identical": reload_predictions_identical,
        "elapsed_seconds": elapsed,
        "formal_paired_control_result_unchanged": True,
        "deployment_checkpoint_only": True,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nStage 3 demonstration checkpoint ready\n")
    print(json.dumps(summary, indent=2))
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Training records: {args.output_dir.resolve()}")
    print(
        "This checkpoint is for the Stage 3 painter. It does not rerun or "
        "replace the paired pixel-control result."
    )


if __name__ == "__main__":
    main()
