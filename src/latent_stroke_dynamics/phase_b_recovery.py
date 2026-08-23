"""Fail-closed lifecycle, environment, and continuity guards for Phase B0 recovery."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import platform
from typing import Any, Mapping

import torch

from .latent_planner import LoadedLatentPredictor, load_formal_latent_predictor
from .phase_b_development import ABORTED_STATUS, load_phase_b_development_config


DEFAULT_RECOVERY_CONFIG = Path("configs/phase-b0-colab-recovery-2026-08-24.json")
DEFAULT_BASE_CONFIG = Path("configs/phase-b-saliency-latent-2026-08-23.json")
PREFLIGHT_REPORT = Path(
    "docs/artifacts/phase-b0-colab-preflight-report-2026-08-24.json"
)
FROZEN_RECOVERY_STATUS = "frozen_before_recovery_implementation"
AUTHORIZED_RECOVERY_STATUS = "recovery_authorized_once"
VALIDATION_STATUS = "phase_b0_colab_recovery_runner_valid_unauthorized"
EXPECTED_MANIFEST_HASHES = {
    "diagnostic_test_transitions.json": "d3101d29de97659a44932282fcbeed807405eecc1f678e71fd36e96a600d997a",
    "planner_supervision.json": "02bef6101b0e380651301bbf7c8c0cf5e02c7c2a39e2dbab13e44fac1a9d186a",
    "train_transitions.json": "7bb572b4d053649d22de75584615441b9d72c014f1a6128b435677e560c6304b",
    "validation_transitions.json": "234ff3b68399aea160ceb0665728d9f1d3d5971e1924ab36eef5c1537558c817",
}
EXPECTED_SOURCE_COMMITS = {
    "stage_a_base": "c211c3ab3a37b9c37eda5ba3c07c01173fd4c7f7",
    "phase_b_protocol": "b3c2c284741dccb413b7811633336edc5e548b26",
    "development_runner": "323df8328e99c26a63fc05194edc43a4ca781efe",
    "interruption_archive": "b93951a5dc3b150199acc66d7b783c6e76dbe88f",
    "colab_preflight_source": "18afc1f3abf301f03417be73d817fea660ef6e45",
    "colab_preflight_archive": "09b4bb1040c495675379a6f2ec9ea43dc74e8fb8",
}
EXPECTED_STAGE_ORDER = (
    "guard_and_environment",
    "resource_integrity",
    "deterministic_data_generation",
    "data_manifest_hash_verification",
    "joint_prediction_only_training_and_checkpoint",
    "joint_prediction_progress_training_and_checkpoint",
    "diagnostics",
    "long_horizon_comparison",
    "eligibility_decision",
    "final_integrity_manifest",
    "atomic_finalize",
)


@dataclass(frozen=True)
class RecoveryOutputPaths:
    final: Path
    incomplete: Path
    journal: Path


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_recovery_config(
    path: str | Path = DEFAULT_RECOVERY_CONFIG,
) -> dict[str, Any]:
    config = _load_json(Path(path))
    validate_recovery_config(config)
    return config


def validate_recovery_config(config: Mapping[str, Any]) -> None:
    required_keys = {
        "experiment_id",
        "status",
        "evidential_role",
        "base_phase_b_experiment_id",
        "branch",
        "scientific_protocol_unchanged",
        "historical_results_unchanged",
        "closed_targets_may_be_reused",
        "source_commits",
        "interrupted_attempt",
        "passing_preflight",
        "environment",
        "device_allocation",
        "data_manifest_sha256_required_before_training",
        "resource_policy",
        "scientific_settings",
        "persistence",
        "required_stage_order",
        "recovery",
        "formal_reserved",
        "phase_b1_reserved",
        "phase_b2_reserved",
        "implementation_validation_boundary",
    }
    if set(config) != required_keys:
        raise ValueError("Phase B0 recovery config fields changed.")
    if config.get("experiment_id") != "phase-b0-colab-cuda-recovery-2026-08-24":
        raise ValueError("Unexpected Phase B0 recovery experiment_id.")
    if config.get("status") != FROZEN_RECOVERY_STATUS:
        raise ValueError("Recovery implementation must remain frozen and unauthorized.")
    if config.get("evidential_role") != (
        "infrastructure_recovery_of_zero-completion_development_attempt"
    ):
        raise ValueError("Recovery evidential role changed.")
    if config.get("base_phase_b_experiment_id") != (
        "phase-b0-action-conditioned-joint-embedding-2026-08-23"
    ):
        raise ValueError("Recovery base experiment changed.")
    if config.get("branch") != "phase-b/saliency-latent":
        raise ValueError("Recovery branch changed.")
    if config.get("scientific_protocol_unchanged") is not True:
        raise ValueError("The scientific protocol must remain unchanged.")
    if config.get("historical_results_unchanged") is not True:
        raise ValueError("Historical results must remain unchanged.")
    if config.get("closed_targets_may_be_reused") is not False:
        raise ValueError("Closed targets may not be reused.")
    if dict(_mapping(config.get("source_commits"), "source_commits")) != EXPECTED_SOURCE_COMMITS:
        raise ValueError("Recovery source commits changed.")
    if dict(
        _mapping(
            config.get("data_manifest_sha256_required_before_training"),
            "data manifest hashes",
        )
    ) != EXPECTED_MANIFEST_HASHES:
        raise ValueError("Recovery data-manifest hashes changed.")
    if tuple(config.get("required_stage_order", ())) != EXPECTED_STAGE_ORDER:
        raise ValueError("Recovery stage order changed.")

    interrupted = _mapping(config.get("interrupted_attempt"), "interrupted_attempt")
    if dict(interrupted) != {
        "record": "configs/phase-b0-aborted-local-attempt-2026-08-23.json",
        "attempts_started": 1,
        "completed_executions": 0,
        "completed_variants": 0,
        "checkpoints_saved": False,
        "scientific_decision_produced": False,
        "local_incomplete_must_remain_untouched": True,
    }:
        raise ValueError("Interrupted-attempt facts changed.")
    preflight = _mapping(config.get("passing_preflight"), "passing_preflight")
    if preflight.get("status") != (
        "phase_b0_colab_cuda_preflight_passed_recovery_unauthorized"
    ):
        raise ValueError("Passing preflight status changed.")
    if preflight.get("local_test_count") != 120 or preflight.get("colab_test_count") != 120:
        raise ValueError("Passing preflight test counts changed.")
    if preflight.get("bundle_sha256") != (
        "80ad2421ac888d239a97a88397402d2233efc5c3b9543f82250fbf0aad0aadbc"
    ):
        raise ValueError("Passing preflight bundle hash changed.")
    if preflight.get("dummy_only") is not True or preflight.get("scientific_evidence") is not False:
        raise ValueError("Preflight evidential boundary changed.")

    environment = _mapping(config.get("environment"), "environment")
    if dict(environment) != {
        "provider": "google_colab_free",
        "device": "cuda:0",
        "gpu_name": "Tesla T4",
        "compute_capability": [7, 5],
        "gpu_memory_bytes": 15637086208,
        "platform": "Linux-6.6.122+-x86_64-with-glibc2.35",
        "python": "3.13.15",
        "torch": "2.11.0+cu128",
        "cuda": "12.8",
        "cudnn": 91900,
        "float_precision": "float32",
        "automatic_mixed_precision": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "matmul_tf32": False,
        "cudnn_tf32": False,
    }:
        raise ValueError("Frozen recovery environment changed.")
    if dict(_mapping(config.get("device_allocation"), "device_allocation")) != {
        "renderer_and_candidate_generation": "cpu",
        "phase_b_training": "cuda:0",
        "phase_b_diagnostics": "cuda:0",
        "phase_b_new_planners": "cuda:0",
        "learned_pixel_planner": "cuda:0",
        "exact_pixel_planner": "cpu",
        "archived_mse_l1_planner": "cpu",
        "historical_checkpoint_integrity_loading": "cpu",
    }:
        raise ValueError("Frozen recovery device allocation changed.")
    if dict(_mapping(config.get("resource_policy"), "resource_policy")) != {
        "raw_and_loaded_state_hashes_must_match_preflight": True,
        "load_task_autoencoder": True,
        "load_mse_only_predictor_seeds": [11, 22, 33],
        "load_pixel_predictor_seed": 11,
        "load_ranking_aware_predictors": False,
        "downloads_allowed": False,
    }:
        raise ValueError("Recovery resource policy changed.")
    if dict(_mapping(config.get("scientific_settings"), "scientific_settings")) != {
        "base_config": "configs/phase-b-saliency-latent-2026-08-23.json",
        "architecture_unchanged": True,
        "objectives_unchanged": True,
        "seeds_unchanged": True,
        "eligibility_thresholds_unchanged": True,
        "method_order_unchanged": True,
        "maximum_completed_recovery_executions": 1,
        "total_wall_clock_cap_hours": 6.0,
    }:
        raise ValueError("Recovery scientific settings changed.")
    if dict(_mapping(config.get("persistence"), "persistence")) != {
        "external_artifact_root_required": True,
        "expected_colab_mount_prefix": "/content/drive/MyDrive",
        "output_leaf": "phase-b0-joint-embedding-development-2026-08-24-colab-recovery",
        "incomplete_suffix": ".incomplete",
        "atomic_incomplete_to_final_rename": True,
        "repository_output_prohibited": True,
        "stage_journal_required": True,
        "save_each_completed_variant_immediately": True,
        "preserve_incomplete_on_error_or_interrupt": True,
        "automatic_resume_authorized": False,
    }:
        raise ValueError("Recovery persistence policy changed.")
    for name in ("recovery", "formal_reserved", "phase_b1_reserved", "phase_b2_reserved"):
        section = _mapping(config.get(name), name)
        if section.get("authorized") is not False:
            raise ValueError(f"{name} must remain unauthorized.")
    if _mapping(config.get("recovery"), "recovery").get("single_run") is not True:
        raise ValueError("Recovery must remain single-run.")
    boundary = _mapping(
        config.get("implementation_validation_boundary"),
        "implementation_validation_boundary",
    )
    forbidden = (
        "may_generate_renderer_transitions",
        "may_generate_targets",
        "may_generate_state_banks",
        "may_generate_candidate_sets",
        "may_create_recovery_output",
        "may_train_on_renderer_data",
        "may_authorize_recovery_or_any_later_phase",
    )
    if any(boundary.get(name) is not False for name in forbidden):
        raise ValueError("Recovery implementation boundary permits a forbidden side effect.")


def recovery_output_paths(
    config: Mapping[str, Any], artifact_root: str | Path
) -> RecoveryOutputPaths:
    persistence = _mapping(config.get("persistence"), "persistence")
    root = Path(artifact_root)
    leaf = str(persistence.get("output_leaf", ""))
    suffix = str(persistence.get("incomplete_suffix", ""))
    if not root.name or not leaf or suffix != ".incomplete":
        raise ValueError("Recovery artifact root or output leaf is invalid.")
    final = root / leaf
    incomplete = root / f"{leaf}{suffix}"
    return RecoveryOutputPaths(
        final=final,
        incomplete=incomplete,
        journal=incomplete / "recovery_stage_journal.json",
    )


def require_recovery_outputs_absent(paths: RecoveryOutputPaths) -> None:
    if paths.final.exists():
        raise FileExistsError(f"Completed recovery output exists: {paths.final}")
    if paths.incomplete.exists():
        raise FileExistsError(
            f"Incomplete recovery output exists: {paths.incomplete}. Preserve and audit it before any retry."
        )


def validate_recovery_runner_request(
    root: str | Path,
    config_path: str | Path = DEFAULT_RECOVERY_CONFIG,
) -> dict[str, Any]:
    repository = Path(root)
    config = load_recovery_config(repository / config_path)
    base = load_phase_b_development_config(repository / DEFAULT_BASE_CONFIG)
    if base.get("status") != ABORTED_STATUS:
        raise ValueError("The zero-completion local attempt is not recovery-locked.")
    if base["development"]["authorized"] is not False:
        raise ValueError("Base Phase B0 development must remain unauthorized.")
    report = _load_json(repository / PREFLIGHT_REPORT)
    if report.get("status") != (
        "phase_b0_colab_cuda_preflight_passed_recovery_unauthorized"
    ):
        raise ValueError("The passing Colab preflight report is missing or changed.")
    if report.get("scientific_models_trained") is not False:
        raise ValueError("The preflight report crossed its scientific boundary.")
    return {
        "status": VALIDATION_STATUS,
        "config_status": config["status"],
        "base_status": base["status"],
        "required_device": config["environment"]["device"],
        "required_gpu": config["environment"]["gpu_name"],
        "expected_data_manifest_hashes": dict(EXPECTED_MANIFEST_HASHES),
        "load_only_mse_predictor_seeds": [11, 22, 33],
        "ranking_aware_models_allowed": False,
        "external_artifact_root_required": True,
        "recovery_authorized": False,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
        "historical_models_loaded": False,
        "renderer_transitions_generated": False,
        "targets_generated": False,
        "state_banks_generated": False,
        "candidate_sets_generated": False,
        "models_trained": False,
        "recovery_output_created": False,
        "local_incomplete_directory_touched": False,
    }


def require_recovery_authorized(
    config: Mapping[str, Any], artifact_root: str | Path
) -> RecoveryOutputPaths:
    recovery = _mapping(config.get("recovery"), "recovery")
    if config.get("status") != AUTHORIZED_RECOVERY_STATUS or recovery.get("authorized") is not True:
        raise PermissionError(
            "Phase B0 Colab recovery is not authorized. Validation must stop before data generation or output creation."
        )
    prefix = str(_mapping(config.get("persistence"), "persistence")["expected_colab_mount_prefix"])
    resolved = str(Path(artifact_root).resolve())
    if resolved != prefix and not resolved.startswith(prefix + "/"):
        raise PermissionError("Recovery artifacts must be written under the frozen Google Drive root.")
    paths = recovery_output_paths(config, artifact_root)
    require_recovery_outputs_absent(paths)
    return paths


def configure_recovery_determinism() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def recovery_environment_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The frozen Phase B0 recovery requires CUDA.")
    device = torch.device("cuda:0")
    return {
        "provider": "google_colab_free",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "float_precision": "float32",
        "automatic_mixed_precision": False,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def validate_recovery_environment_snapshot(
    config: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> None:
    expected = dict(_mapping(config.get("environment"), "environment"))
    if dict(snapshot) != expected:
        changed = sorted(
            key for key in set(expected) | set(snapshot) if expected.get(key) != snapshot.get(key)
        )
        raise RuntimeError(
            "Frozen Colab recovery environment mismatch: " + ", ".join(changed)
        )


def validate_expected_data_manifests(
    config: Mapping[str, Any], data_root: str | Path
) -> dict[str, str]:
    expected = dict(
        _mapping(
            config.get("data_manifest_sha256_required_before_training"),
            "data manifest hashes",
        )
    )
    root = Path(data_root)
    actual: dict[str, str] = {}
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required recovery data manifest is missing: {path}")
        actual[name] = file_sha256(path)
        if actual[name] != digest:
            raise RuntimeError(f"Recovery data manifest SHA-256 mismatch: {name}")
    if set(path.name for path in root.glob("*.json")) != set(expected):
        raise RuntimeError("Recovery data-manifest directory contains unexpected JSON files.")
    return actual


def load_recovery_mse_only_predictors(
    closed_config: Mapping[str, Any],
) -> tuple[LoadedLatentPredictor, ...]:
    predictors = _mapping(closed_config.get("latent_predictors"), "latent_predictors")
    entries = predictors.get("mse_only")
    if not isinstance(entries, list) or len(entries) != 3:
        raise ValueError("Recovery requires exactly three MSE-only predictors.")
    loaded = tuple(
        load_formal_latent_predictor(
            entry["path"],
            expected_method="mse_only",
            expected_seed=int(entry["seed"]),
            expected_state_sha256=entry["state_sha256"],
        )
        for entry in entries
    )
    if tuple(item.seed for item in loaded) != (11, 22, 33):
        raise RuntimeError("Recovery MSE-only predictor seed order changed.")
    return loaded
