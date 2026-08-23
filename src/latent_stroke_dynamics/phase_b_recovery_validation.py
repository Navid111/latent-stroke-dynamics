"""Dummy-only CUDA validation for the guarded Phase B0 recovery runner."""

from __future__ import annotations

import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import torch

from .extension_training import model_state_sha256
from .phase_b_cloud_preflight import (
    compare_cpu_cuda_outputs,
    verify_loaded_model_states,
    verify_raw_resources,
)
from .phase_b_data import (
    PlannerCandidateSet,
    PlannerTensorPayload,
    TransitionTensorPayload,
)
from .phase_b_planning import phase_b_candidate_scores
from .phase_b_recovery import (
    configure_recovery_determinism,
    load_recovery_config,
    recovery_environment_snapshot,
    validate_recovery_environment_snapshot,
    validate_recovery_runner_request,
)
from .phase_b_training import (
    feature_statistics,
    freeze_phase_b_model,
    planner_candidate_metrics,
    save_phase_b_checkpoint,
    train_phase_b_variant,
)
from .renderer import Stroke


VALIDATION_STATUS = "phase_b0_colab_recovery_implementation_valid_unauthorized"


def validate_cuda_recovery_boundary(root: str | Path = ".") -> dict[str, Any]:
    result = validate_recovery_runner_request(root)
    if result["recovery_authorized"] is not False:
        raise PermissionError("Dummy CUDA validation requires recovery unauthorized.")
    return {
        "status": "phase_b0_colab_recovery_cuda_boundary_valid_unauthorized",
        "runner_status": result["status"],
        "recovery_authorized": False,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
        "dummy_tensors_only": True,
        "temporary_dummy_checkpoints_allowed": True,
        "renderer_transitions_allowed": False,
        "targets_allowed": False,
        "state_banks_allowed": False,
        "candidate_sets_allowed": False,
        "scientific_training_allowed": False,
        "recovery_output_allowed": False,
    }


def _dummy_payloads() -> tuple[
    TransitionTensorPayload,
    TransitionTensorPayload,
    PlannerTensorPayload,
]:
    generator = torch.Generator(device="cpu").manual_seed(20260825)
    current = torch.rand(2, 1, 64, 64, generator=generator)
    next_canvas = torch.rand(2, 1, 64, 64, generator=generator)
    actions = torch.rand(2, 2, 64, 64, generator=generator)
    actions[0].zero_()
    transition = TransitionTensorPayload(
        examples=(),
        current=current,
        next_canvas=next_canvas,
        actions=actions,
        no_op=torch.tensor([True, False]),
    )
    candidate_count = 32
    planner_current = torch.rand(candidate_count, 1, 64, 64, generator=generator)
    planner_next = torch.rand(candidate_count, 1, 64, 64, generator=generator)
    planner_target = torch.rand(candidate_count, 1, 64, 64, generator=generator)
    planner_actions = torch.rand(candidate_count, 2, 64, 64, generator=generator)
    planner_actions[0].zero_()
    exact_progress = torch.linspace(-0.02, 0.03, candidate_count)
    exact_progress[0] = 0.0
    blank = Image.new("L", (64, 64), 255)
    record = PlannerCandidateSet(
        set_id=0,
        target_seed=-1,
        trajectory_seed=-1,
        candidate_seed=-1,
        state_name="dummy_only",
        current=blank,
        target=blank,
        candidates=(None,) * candidate_count,
    )
    planner = PlannerTensorPayload(
        records=(record,),
        current=planner_current,
        next_canvas=planner_next,
        target=planner_target,
        actions=planner_actions,
        exact_progress=exact_progress,
        set_index=torch.zeros(candidate_count, dtype=torch.int64),
        candidate_index=torch.arange(candidate_count, dtype=torch.int64),
    )
    return transition, transition, planner


def dummy_recovery_execution_smoke(
    device: str | torch.device,
) -> dict[str, Any]:
    """Exercise modified training, diagnostics, planning scores, and checkpoint I/O."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA dummy recovery validation requested without CUDA.")
    train, validation, planner = _dummy_payloads()
    fits: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    temporary_path: Path | None = None
    with tempfile.TemporaryDirectory(prefix="phase-b0-recovery-validation-") as temporary:
        temporary_path = Path(temporary)
        for variant in ("joint_prediction_only", "joint_prediction_progress"):
            fit = train_phase_b_variant(
                variant,
                train,
                validation,
                planner,
                planner,
                progress_mean=0.0,
                progress_std=0.02,
                seed=73,
                learning_rate=0.0003,
                weight_decay=0.0001,
                batch_size=2,
                maximum_epochs=1,
                patience=1,
                gradient_clip_norm=5.0,
                wall_clock_cap_hours=1.0,
                device=resolved,
            )
            freeze_phase_b_model(fit.model)
            checkpoint, digest = save_phase_b_checkpoint(
                fit,
                temporary_path / f"{variant}.pt",
                progress_mean=0.0,
                progress_std=0.02,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["state_sha256"] != digest:
                raise RuntimeError("Dummy recovery checkpoint state hash changed after save.")
            if model_state_sha256(fit.model) != digest:
                raise RuntimeError("Dummy recovery in-memory and saved hashes differ.")
            fits[variant] = fit
            checkpoints[variant] = {
                "best_epoch": fit.best_epoch,
                "best_validation_loss": fit.best_validation_loss,
                "training_device": fit.training_device,
                "state_sha256": digest,
                "temporary_checkpoint_exists_during_validation": checkpoint.is_file(),
            }

        representation = {
            variant: feature_statistics(fit.model, validation)
            for variant, fit in fits.items()
        }
        candidate_metrics = planner_candidate_metrics(
            fits["joint_prediction_only"].model, planner, "prediction"
        ) + planner_candidate_metrics(
            fits["joint_prediction_progress"].model, planner, "progress"
        )
        dummy_current = Image.new("L", (64, 64), 255)
        dummy_target = Image.new("L", (64, 64), 128)
        dummy_strokes = (
            Stroke(0.1, 0.1, 0.9, 0.9, 1, 0),
            Stroke(0.1, 0.9, 0.9, 0.1, 2, 32),
            Stroke(0.2, 0.5, 0.8, 0.5, 3, 64),
            Stroke(0.5, 0.2, 0.5, 0.8, 4, 96),
        )
        prediction_scores = phase_b_candidate_scores(
            fits["joint_prediction_only"].model,
            dummy_current,
            dummy_target,
            dummy_strokes,
            mode="prediction",
            batch_size=2,
        )
        progress_scores = phase_b_candidate_scores(
            fits["joint_prediction_progress"].model,
            dummy_current,
            dummy_target,
            dummy_strokes,
            mode="progress",
            batch_size=2,
        )
        numeric = [
            item["best_validation_loss"] for item in checkpoints.values()
        ] + prediction_scores.tolist() + progress_scores.tolist()
        if not all(math.isfinite(float(value)) for value in numeric):
            raise RuntimeError("Dummy recovery validation produced a non-finite value.")
        peak_memory = (
            int(torch.cuda.max_memory_allocated(resolved))
            if resolved.type == "cuda"
            else 0
        )
    assert temporary_path is not None
    return {
        "device": str(resolved),
        "variants": checkpoints,
        "representation": representation,
        "planner_candidate_metric_rows": len(candidate_metrics),
        "prediction_score_count": len(prediction_scores),
        "progress_score_count": len(progress_scores),
        "all_values_finite": True,
        "temporary_dummy_checkpoints_removed": not temporary_path.exists(),
        "peak_cuda_memory_bytes": peak_memory,
        "dummy_tensors_only": True,
        "scientific_evidence": False,
    }


def run_cuda_recovery_validation(root: str | Path = ".") -> dict[str, Any]:
    boundary = validate_cuda_recovery_boundary(root)
    config = load_recovery_config(Path(root) / "configs/phase-b0-colab-recovery-2026-08-24.json")
    configure_recovery_determinism()
    environment = recovery_environment_snapshot()
    validate_recovery_environment_snapshot(config, environment)
    raw_resources = verify_raw_resources(root)
    loaded_states = verify_loaded_model_states(root)
    if loaded_states.get("ranking_aware_models_loaded") is not False:
        raise RuntimeError("Dummy recovery validation loaded a ranking-aware model.")
    numerical = compare_cpu_cuda_outputs()
    if numerical.get("passed") is not True:
        raise RuntimeError("Dummy recovery CPU/CUDA numerical check failed.")
    execution = dummy_recovery_execution_smoke("cuda:0")
    if execution.get("all_values_finite") is not True:
        raise RuntimeError("Dummy recovery execution smoke did not pass.")
    return {
        "status": VALIDATION_STATUS,
        "boundary": boundary,
        "environment": environment,
        "raw_resource_sha256": raw_resources,
        "loaded_model_states": loaded_states,
        "cpu_cuda_numerical_check": numerical,
        "dummy_recovery_execution_smoke": execution,
        "recovery_authorized": False,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
        "renderer_transitions_generated": False,
        "targets_generated": False,
        "state_banks_generated": False,
        "candidate_sets_generated": False,
        "scientific_models_trained": False,
        "recovery_output_created": False,
        "temporary_dummy_checkpoints_removed": True,
        "dummy_metrics_are_scientific_evidence": False,
    }
