"""Dummy-only CUDA and frozen-resource checks for Phase B0 cloud recovery."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any

import torch

from .extension_training import model_state_sha256
from .latent_planner import (
    load_formal_latent_predictor,
    load_latent_planner_config,
    load_task_latent_resources,
)
from .learned_pixel_planner import load_pixel_checkpoint, state_dict_sha256
from .phase_b_development import (
    ABORTED_STATUS,
    DEFAULT_PHASE_B_CONFIG,
    EXPECTED_ABORTED_AUTHORIZATION,
    load_phase_b_development_config,
    phase_b_output_paths,
)
from .phase_b_joint_embedding import (
    MultiScaleActionJointEmbeddingModel,
    phase_b_objective,
    trainable_parameter_count,
)


ABORTED_ATTEMPT_RECORD = Path(
    "configs/phase-b0-aborted-local-attempt-2026-08-23.json"
)
LATENT_PLANNER_CONFIG = Path("configs/latent-planner-2026-08-23.json")
FROZEN_RESOURCE_RAW_HASHES = {
    "outputs/representation-extension-2026-08-22/task_autoencoder/checkpoints/task_autoencoder.pt": "65debde397895ff25f0b3bf10d4fc8d2c47487ec13887d03be602195b328696d",
    "outputs/representation-extension-2026-08-22/task_autoencoder/latent_channel_statistics.json": "c2a3d781dab19a4714189d580dafb5ea95231af06021d3980beb495a3b85d903",
    "outputs/ranking-aware-latent-formal-2026-08-22/checkpoints/mse_only_seed11.pt": "f14ae080d7ac4a34e536eab706405d79390b17efafbcb0201177a01ea99b886d",
    "outputs/ranking-aware-latent-formal-2026-08-22/checkpoints/mse_only_seed22.pt": "cf5e0c7808fb8d2a388e186d4803f17a05f81d9f8ebcf0fe54ebe37929cdb46a",
    "outputs/ranking-aware-latent-formal-2026-08-22/checkpoints/mse_only_seed33.pt": "f72e941c0d7157347327282311cd14cbc64c7ed0639ff8d807a0e947e80981dc",
    "checkpoints/stage3-pixel-mlp-seed11.pt": "7b9fe1598ee50bcc79bd7f5e9762aff6eb3183cc1266adf5541627cbdbc4e4eb",
}
EXPECTED_TRAINABLE_PARAMETERS = 392_345
CPU_CUDA_ABSOLUTE_TOLERANCE = 5e-4
CPU_CUDA_RELATIVE_TOLERANCE = 5e-4


def raw_file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_resource_paths() -> tuple[str, ...]:
    """Return exactly the six used resources, excluding ranking-aware models."""

    paths = tuple(FROZEN_RESOURCE_RAW_HASHES)
    if any("ranking_aware_seed" in path for path in paths):
        raise RuntimeError("Cloud recovery must not require unused ranking-aware models.")
    return paths


def validate_cloud_preflight_boundary(root: str | Path = ".") -> dict[str, Any]:
    """Confirm recovery is locked before resource loading or CUDA work."""

    root_path = Path(root).resolve()
    config = load_phase_b_development_config(root_path / DEFAULT_PHASE_B_CONFIG)
    paths = phase_b_output_paths(config)
    if config["status"] != ABORTED_STATUS:
        raise PermissionError("Cloud preflight requires the archived interrupted status.")
    if config["development"]["authorized"] is not False:
        raise PermissionError("Phase B0 development must remain unauthorized.")
    if config["development_authorization"] != EXPECTED_ABORTED_AUTHORIZATION:
        raise ValueError("The interrupted-attempt lifecycle record changed.")
    if (root_path / paths.final).exists():
        raise FileExistsError("A completed Phase B0 development output already exists.")
    if not (root_path / ABORTED_ATTEMPT_RECORD).is_file():
        raise FileNotFoundError("The interrupted-attempt audit record is missing.")
    return {
        "status": "phase_b0_cloud_preflight_boundary_valid_recovery_unauthorized",
        "config_status": config["status"],
        "development_authorized": False,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
        "renderer_data_generation_allowed": False,
        "scientific_training_allowed": False,
        "dummy_tensors_only": True,
        "local_incomplete_directory_may_exist": True,
    }


def verify_raw_resources(root: str | Path = ".") -> dict[str, str]:
    root_path = Path(root).resolve()
    observed: dict[str, str] = {}
    for relative, expected in FROZEN_RESOURCE_RAW_HASHES.items():
        path = root_path / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen cloud resource: {relative}")
        digest = raw_file_sha256(path)
        if digest != expected:
            raise ValueError(f"Raw SHA-256 mismatch for frozen resource: {relative}")
        observed[relative] = digest
    return observed


def verify_loaded_model_states(root: str | Path = ".") -> dict[str, Any]:
    """Load only the resources actually used by the frozen Phase B0 comparison."""

    root_path = Path(root).resolve()
    closed = load_latent_planner_config(root_path / LATENT_PLANNER_CONFIG)
    portable = deepcopy(closed)
    portable["representation"]["autoencoder_checkpoint"] = str(
        root_path / closed["representation"]["autoencoder_checkpoint"]
    )
    portable["representation"]["latent_statistics"] = str(
        root_path / closed["representation"]["latent_statistics"]
    )
    autoencoder, _ = load_task_latent_resources(portable)
    autoencoder_digest = model_state_sha256(autoencoder)

    mse_digests: dict[str, str] = {}
    for entry in closed["latent_predictors"]["mse_only"]:
        loaded = load_formal_latent_predictor(
            root_path / entry["path"],
            expected_method="mse_only",
            expected_seed=int(entry["seed"]),
            expected_state_sha256=str(entry["state_sha256"]),
        )
        mse_digests[str(entry["seed"])] = loaded.state_sha256

    pixel = closed["pixel_predictor"]
    pixel_model, metadata = load_pixel_checkpoint(
        root_path / pixel["path"], device="cpu"
    )
    pixel_digest = state_dict_sha256(pixel_model)
    if pixel_digest != pixel["state_sha256"]:
        raise ValueError("Frozen pixel predictor state SHA-256 mismatch.")
    if metadata.model_seed != 11:
        raise ValueError("Frozen pixel predictor seed changed.")
    return {
        "task_autoencoder_state_sha256": autoencoder_digest,
        "mse_only_state_sha256": mse_digests,
        "pixel_predictor_state_sha256": pixel_digest,
        "pixel_predictor_model_seed": metadata.model_seed,
        "ranking_aware_models_loaded": False,
    }


def environment_report() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        report.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_compute_capability": list(torch.cuda.get_device_capability(0)),
                "cuda_total_memory_bytes": int(properties.total_memory),
            }
        )
    return report


def _configure_determinism() -> None:
    torch.manual_seed(73)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(73)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def compare_cpu_cuda_outputs() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Colab GPU runtime.")
    _configure_determinism()
    cpu_model = MultiScaleActionJointEmbeddingModel().cpu().eval()
    cuda_model = MultiScaleActionJointEmbeddingModel().cuda().eval()
    cuda_model.load_state_dict(cpu_model.state_dict(), strict=True)
    generator = torch.Generator().manual_seed(20260824)
    current = torch.rand(4, 1, 64, 64, generator=generator)
    actions = torch.rand(4, 2, 64, 64, generator=generator)
    actions[0].zero_()
    goal = torch.rand(4, 1, 64, 64, generator=generator)
    with torch.inference_mode():
        cpu = cpu_model(current, actions, goal)
        cuda = cuda_model(current.cuda(), actions.cuda(), goal.cuda())
    pairs = {
        "current_32": (cpu["current"]["32"], cuda["current"]["32"]),
        "current_16": (cpu["current"]["16"], cuda["current"]["16"]),
        "action_32": (cpu["action"]["32"], cuda["action"]["32"]),
        "action_16": (cpu["action"]["16"], cuda["action"]["16"]),
        "predicted_next_32": (
            cpu["predicted_next"]["32"],
            cuda["predicted_next"]["32"],
        ),
        "predicted_next_16": (
            cpu["predicted_next"]["16"],
            cuda["predicted_next"]["16"],
        ),
        "predicted_progress": (
            cpu["predicted_progress"],
            cuda["predicted_progress"],
        ),
    }
    errors: dict[str, float] = {}
    passed = True
    for name, (cpu_value, cuda_value) in pairs.items():
        observed = cuda_value.detach().cpu()
        errors[name] = float((cpu_value - observed).abs().max().item())
        passed = passed and bool(
            torch.allclose(
                cpu_value,
                observed,
                atol=CPU_CUDA_ABSOLUTE_TOLERANCE,
                rtol=CPU_CUDA_RELATIVE_TOLERANCE,
            )
        )
    return {
        "passed": passed,
        "absolute_tolerance": CPU_CUDA_ABSOLUTE_TOLERANCE,
        "relative_tolerance": CPU_CUDA_RELATIVE_TOLERANCE,
        "maximum_absolute_error": max(errors.values()),
        "per_output_maximum_absolute_error": errors,
    }


def _dummy_batch(
    batch_size: int, device: torch.device, seed: int
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    current = torch.rand(batch_size, 1, 64, 64, generator=generator, device=device)
    next_canvas = torch.rand(batch_size, 1, 64, 64, generator=generator, device=device)
    actions = torch.rand(batch_size, 2, 64, 64, generator=generator, device=device)
    actions[0].zero_()
    goal = torch.rand(batch_size, 1, 64, 64, generator=generator, device=device)
    exact_progress = torch.linspace(-0.02, 0.03, batch_size, device=device)
    exact_progress[0] = 0.0
    return {
        "current": current,
        "next_canvas": next_canvas,
        "actions": actions,
        "goal": goal,
        "no_op": torch.arange(batch_size, device=device).eq(0),
        "exact_progress": exact_progress,
    }


def _optimizer_step(
    model: MultiScaleActionJointEmbeddingModel,
    optimizer: torch.optim.Optimizer,
    variant: str,
    batch: dict[str, torch.Tensor],
) -> float:
    if variant == "joint_prediction_only":
        output = model(batch["current"], batch["actions"])
        target = model.encode_target(batch["next_canvas"])
        losses = phase_b_objective(
            variant=variant,
            online_features=output["current"],
            predicted_next=output["predicted_next"],
            target_next=target,
            residuals=output["residual"],
            action_rasters=batch["actions"],
            no_op_examples=batch["no_op"],
        )
    elif variant == "joint_prediction_progress":
        output = model(batch["current"], batch["actions"], batch["goal"])
        target = model.encode_target(batch["next_canvas"])
        losses = phase_b_objective(
            variant=variant,
            online_features=output["current"],
            predicted_next=output["predicted_next"],
            target_next=target,
            residuals=output["residual"],
            action_rasters=batch["actions"],
            no_op_examples=batch["no_op"],
            predicted_progress=output["predicted_progress"].reshape(1, -1),
            exact_progress=batch["exact_progress"].reshape(1, -1),
            progress_training_mean=0.0,
            progress_training_std=0.02,
        )
    else:
        raise ValueError("Unexpected dummy benchmark variant.")

    loss = losses["total"]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise RuntimeError("Dummy benchmark produced missing or non-finite gradients.")
    if variant == "joint_prediction_progress" and len(gradients) != len(parameters):
        raise RuntimeError("Progress dummy objective did not reach every trainable parameter.")
    torch.nn.utils.clip_grad_norm_(parameters, 5.0)
    optimizer.step()
    model.update_target_encoder(0.99)
    value = float(loss.detach().item())
    if not math.isfinite(value):
        raise RuntimeError("Dummy benchmark produced a non-finite loss.")
    return value


def dummy_optimizer_smoke(device: str | torch.device = "cpu") -> dict[str, Any]:
    resolved = torch.device(device)
    _configure_determinism()
    model = MultiScaleActionJointEmbeddingModel().to(resolved).train()
    if trainable_parameter_count(model) != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError("Phase B0 parameter count changed.")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-4,
        weight_decay=1e-4,
    )
    prediction = _optimizer_step(
        model,
        optimizer,
        "joint_prediction_only",
        _dummy_batch(2, resolved, 8101),
    )
    progress = _optimizer_step(
        model,
        optimizer,
        "joint_prediction_progress",
        _dummy_batch(2, resolved, 8102),
    )
    return {
        "device": str(resolved),
        "joint_prediction_only_dummy_loss": prediction,
        "joint_prediction_progress_dummy_loss": progress,
        "finite": True,
    }


def benchmark_cuda(warmup_steps: int = 2, measured_steps: int = 8) -> dict[str, Any]:
    if warmup_steps < 1 or measured_steps < 1:
        raise ValueError("Benchmark step counts must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select a Colab GPU runtime.")
    _configure_determinism()
    device = torch.device("cuda")
    model = MultiScaleActionJointEmbeddingModel().to(device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=3e-4,
        weight_decay=1e-4,
    )
    transition_batch = _dummy_batch(16, device, 8201)
    planner_batch = _dummy_batch(32, device, 8202)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup_steps):
        _optimizer_step(model, optimizer, "joint_prediction_only", transition_batch)
        _optimizer_step(model, optimizer, "joint_prediction_progress", planner_batch)
    torch.cuda.synchronize()

    timings: dict[str, list[float]] = {
        "joint_prediction_only": [],
        "joint_prediction_progress": [],
    }
    for variant, batch in (
        ("joint_prediction_only", transition_batch),
        ("joint_prediction_progress", planner_batch),
    ):
        for _ in range(measured_steps):
            torch.cuda.synchronize()
            started = time.perf_counter()
            _optimizer_step(model, optimizer, variant, batch)
            torch.cuda.synchronize()
            timings[variant].append((time.perf_counter() - started) * 1000.0)

    transition_ms = statistics.median(timings["joint_prediction_only"])
    planner_ms = statistics.median(timings["joint_prediction_progress"])
    transition_steps = 2 * 40 * math.ceil(2048 / 16)
    planner_steps = 40 * 64
    training_seconds = (
        transition_ms * transition_steps + planner_ms * planner_steps
    ) / 1000.0
    return {
        "warmup_steps_per_variant": warmup_steps,
        "measured_steps_per_variant": measured_steps,
        "transition_batch_size": 16,
        "planner_candidate_batch_size": 32,
        "median_transition_optimizer_step_ms": transition_ms,
        "median_planner_optimizer_step_ms": planner_ms,
        "maximum_epoch_transition_optimizer_steps": transition_steps,
        "maximum_progress_optimizer_steps": planner_steps,
        "estimated_maximum_training_minutes_excluding_validation": training_seconds / 60.0,
        "three_x_safety_training_hours_excluding_validation": 3.0 * training_seconds / 3600.0,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "dummy_tensors_only": True,
    }


def run_colab_preflight(root: str | Path = ".") -> dict[str, Any]:
    boundary = validate_cloud_preflight_boundary(root)
    environment = environment_report()
    if not environment["cuda_available"]:
        raise RuntimeError("CUDA is unavailable. Select Runtime > Change runtime type > GPU.")
    raw = verify_raw_resources(root)
    states = verify_loaded_model_states(root)
    numerical = compare_cpu_cuda_outputs()
    if not numerical["passed"]:
        raise RuntimeError("CPU/CUDA dummy-output tolerance check failed.")
    smoke = dummy_optimizer_smoke("cuda")
    benchmark = benchmark_cuda()
    return {
        "status": "phase_b0_colab_cuda_preflight_passed_recovery_unauthorized",
        "boundary": boundary,
        "environment": environment,
        "raw_resource_sha256": raw,
        "loaded_model_states": states,
        "cpu_cuda_numerical_check": numerical,
        "dummy_optimizer_smoke": smoke,
        "dummy_cuda_benchmark": benchmark,
        "renderer_transitions_generated": False,
        "targets_generated": False,
        "state_banks_generated": False,
        "candidate_sets_generated": False,
        "scientific_models_trained": False,
        "scientific_outputs_created": False,
        "recovery_authorized": False,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
        "dummy_metrics_are_scientific_evidence": False,
    }
