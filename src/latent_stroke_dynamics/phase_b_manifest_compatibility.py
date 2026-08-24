"""Pinned, manifest-only compatibility gate for the consumed Phase B0 recovery."""

from __future__ import annotations

import gc
from hashlib import sha256
import json
from pathlib import Path
import platform
import time
from typing import Any, Mapping

import numpy as np
import PIL
import torch

from .phase_b_data import (
    build_planner_payload,
    build_transition_payload,
    fit_progress_statistics,
)
from .phase_b_development import ABORTED_STATUS, load_phase_b_development_config


DEFAULT_COMPATIBILITY_CONFIG = Path(
    "configs/phase-b0-colab-manifest-compatibility-2026-08-24.json"
)
FROZEN_COMPATIBILITY_STATUS = "manifest_only_compatibility_frozen_unauthorized"
PASS_STATUS = "phase_b0_colab_manifest_compatibility_passed_unauthorized"
FAIL_STATUS = "phase_b0_colab_manifest_compatibility_failed_unauthorized"
EXPECTED_ORIGINAL_MANIFEST_HASHES = {
    "diagnostic_test_transitions.json": "d3101d29de97659a44932282fcbeed807405eecc1f678e71fd36e96a600d997a",
    "planner_supervision.json": "02bef6101b0e380651301bbf7c8c0cf5e02c7c2a39e2dbab13e44fac1a9d186a",
    "train_transitions.json": "7bb572b4d053649d22de75584615441b9d72c014f1a6128b435677e560c6304b",
    "validation_transitions.json": "234ff3b68399aea160ceb0665728d9f1d3d5971e1924ab36eef5c1537558c817",
}


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


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest_compatibility_config(config: Mapping[str, Any]) -> None:
    required_keys = {
        "experiment_id",
        "status",
        "purpose",
        "branch",
        "source_incident_commit",
        "source_consumed_attempt",
        "base_config",
        "historical_evidence_unchanged",
        "expected_original_manifest_sha256",
        "required_environment",
        "generation",
        "output_policy",
        "validation_boundary",
        "post_validation",
    }
    if set(config) != required_keys:
        raise ValueError("Manifest-compatibility config fields changed.")
    if config.get("experiment_id") != (
        "phase-b0-colab-manifest-compatibility-2026-08-24"
    ):
        raise ValueError("Manifest-compatibility experiment changed.")
    if config.get("status") != FROZEN_COMPATIBILITY_STATUS:
        raise ValueError("Manifest compatibility must remain frozen and unauthorized.")
    if config.get("branch") != "phase-b/saliency-latent":
        raise ValueError("Manifest-compatibility branch changed.")
    if config.get("historical_evidence_unchanged") is not True:
        raise ValueError("Historical evidence must remain unchanged.")
    if dict(
        _mapping(
            config.get("expected_original_manifest_sha256"),
            "expected manifest hashes",
        )
    ) != EXPECTED_ORIGINAL_MANIFEST_HASHES:
        raise ValueError("Original manifest hashes changed.")
    if dict(_mapping(config.get("required_environment"), "required environment")) != {
        "numpy": "2.5.2",
        "pillow": "12.3.0",
        "torch_base": "2.11.0",
        "generation_device": "cpu",
        "renderer_boundary": "PIL.ImageDraw.line",
    }:
        raise ValueError("Pinned manifest-compatibility environment changed.")
    generation = _mapping(config.get("generation"), "generation")
    if tuple(generation.get("manifest_filenames", ())) != tuple(
        EXPECTED_ORIGINAL_MANIFEST_HASHES
    ):
        raise ValueError("Manifest filename set or order changed.")
    if generation.get("persist_images_or_tensors") is not False:
        raise ValueError("Compatibility validation may not persist images or tensors.")
    if generation.get("maximum_attempts_per_fresh_runtime") != 1:
        raise ValueError("Compatibility validation permits one attempt per fresh runtime.")
    output = _mapping(config.get("output_policy"), "output policy")
    for name in (
        "fresh_output_directory_required",
        "repository_output_prohibited",
        "google_drive_output_prohibited",
        "preserve_generated_manifests_on_hash_mismatch",
    ):
        if output.get(name) is not True:
            raise ValueError(f"Output policy {name} changed.")
    if output.get("touch_historical_incomplete_directories") is not False:
        raise ValueError("Historical incomplete directories must remain untouched.")
    boundary = _mapping(config.get("validation_boundary"), "validation boundary")
    if boundary.get("renderer_manifest_generation_allowed") is not True:
        raise ValueError("Manifest generation must be the only active validation path.")
    if boundary.get("in_memory_targets_states_and_candidates_allowed") is not True:
        raise ValueError("The frozen planner manifests require in-memory state generation.")
    forbidden = (
        "model_resource_loading_allowed",
        "checkpoint_creation_allowed",
        "scientific_training_allowed",
        "recovery_execution_allowed",
        "decision_creation_allowed",
        "formal_authorization_allowed",
        "phase_b1_authorization_allowed",
        "phase_b2_authorization_allowed",
    )
    if any(boundary.get(name) is not False for name in forbidden):
        raise ValueError("Manifest compatibility permits a forbidden side effect.")
    post = _mapping(config.get("post_validation"), "post validation")
    if post.get("all_four_hashes_required") is not True:
        raise ValueError("All four original hashes must be required.")
    if post.get("passing_result_is_scientific_training_authorization") is not False:
        raise ValueError("A compatibility result may not authorize training.")
    if post.get("old_authorization_reusable") is not False:
        raise ValueError("The consumed authorization may not be reused.")


def load_manifest_compatibility_config(
    path: str | Path = DEFAULT_COMPATIBILITY_CONFIG,
) -> dict[str, Any]:
    config = _load_json(Path(path))
    validate_manifest_compatibility_config(config)
    return config


def load_consumed_attempt_record(
    repository_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(repository_root) / str(config["source_consumed_attempt"])
    record = _load_json(path)
    expected = {
        "status": "phase_b0_colab_recovery_attempt_consumed_manifest_mismatch",
        "execution_attempt_consumed": True,
        "rerun_authorized": False,
        "failure_stage": "data_manifest_hash_verification",
        "training_started": False,
        "models_trained": False,
        "next_allowed_action": "new_unauthorized_manifest_only_compatibility_protocol",
    }
    changed = [name for name, value in expected.items() if record.get(name) != value]
    if changed:
        raise ValueError("Consumed recovery record changed: " + ", ".join(changed))
    return record


def runtime_environment_snapshot() -> dict[str, Any]:
    torch_version = str(torch.__version__)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "torch": torch_version,
        "torch_base": torch_version.split("+", 1)[0],
        "cuda_available": bool(torch.cuda.is_available()),
        "generation_device": "cpu",
        "renderer_boundary": "PIL.ImageDraw.line",
    }


def validate_required_environment(
    config: Mapping[str, Any], snapshot: Mapping[str, Any]
) -> None:
    required = _mapping(config.get("required_environment"), "required environment")
    checks = {
        "numpy": required["numpy"],
        "pillow": required["pillow"],
        "torch_base": required["torch_base"],
        "generation_device": required["generation_device"],
        "renderer_boundary": required["renderer_boundary"],
    }
    changed = [name for name, value in checks.items() if snapshot.get(name) != value]
    if changed:
        raise RuntimeError(
            "Pinned manifest-compatibility environment mismatch: "
            + ", ".join(changed)
        )


def guard_output_directory(
    output_dir: str | Path,
    repository_root: str | Path,
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    repository = Path(repository_root).resolve()
    if output == repository or repository in output.parents:
        raise PermissionError("Manifest compatibility output may not be inside the repository.")
    drive = Path("/content/drive").resolve()
    if output == drive or drive in output.parents:
        raise PermissionError("Manifest compatibility may not write to Google Drive.")
    if output.exists():
        raise FileExistsError(
            f"Manifest compatibility output already exists: {output}. Use a fresh runtime or a new audited path."
        )
    return output


def compare_manifest_hashes(
    data_root: str | Path,
    expected: Mapping[str, str],
) -> dict[str, Any]:
    root = Path(data_root)
    expected_names = set(expected)
    actual_names = {path.name for path in root.glob("*.json") if path.is_file()}
    files: dict[str, Any] = {}
    for name, expected_digest in expected.items():
        path = root / name
        actual_digest = file_sha256(path) if path.is_file() else None
        files[name] = {
            "expected_sha256": expected_digest,
            "actual_sha256": actual_digest,
            "match": actual_digest == expected_digest,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    names_match = actual_names == expected_names
    return {
        "files": files,
        "expected_filenames": sorted(expected_names),
        "actual_filenames": sorted(actual_names),
        "filename_set_matches": names_match,
        "all_hashes_match": names_match
        and all(item["match"] for item in files.values()),
    }


def _transition_manifest(payload: Any, split_name: str, seed: int) -> dict[str, Any]:
    return {
        "split": split_name,
        "seed": seed,
        "samples": payload.size,
        "no_op_samples": int(payload.no_op.sum().item()),
        "fingerprints": [item.fingerprint for item in payload.examples],
    }


def _planner_records(payload: Any) -> list[dict[str, Any]]:
    return [
        {
            "set_id": item.set_id,
            "target_seed": item.target_seed,
            "trajectory_seed": item.trajectory_seed,
            "candidate_seed": item.candidate_seed,
            "state_name": item.state_name,
        }
        for item in payload.records
    ]


def generate_manifest_only_data(
    base_config: Mapping[str, Any],
    data_root: str | Path,
) -> dict[str, Any]:
    """Generate only the four historical JSON manifests, sequentially on CPU."""

    root = Path(data_root)
    root.mkdir(parents=True)
    crowding = tuple(
        int(value) for value in base_config["renderer"]["transition_crowding"]
    )
    transition_summary: dict[str, Any] = {}
    for split_name, definition in base_config["development"][
        "transition_splits"
    ].items():
        payload = build_transition_payload(
            samples=int(definition["samples"]),
            seed=int(definition["seed"]),
            crowding_levels=crowding,
            no_op_fraction=float(
                base_config["objectives"]["no_op_consistency"][
                    "transition_fraction"
                ]
            ),
        )
        _write_json_atomic(
            root / f"{split_name}_transitions.json",
            _transition_manifest(payload, split_name, int(definition["seed"])),
        )
        transition_summary[split_name] = {
            "samples": payload.size,
            "no_op_samples": int(payload.no_op.sum().item()),
        }
        del payload
        gc.collect()

    planner_train = build_planner_payload(base_config, "planner_supervision_train")
    progress_mean, progress_std = fit_progress_statistics(planner_train)
    training_sets = planner_train.candidate_sets
    training_records = _planner_records(planner_train)
    del planner_train
    gc.collect()

    planner_validation = build_planner_payload(
        base_config, "planner_supervision_validation"
    )
    validation_sets = planner_validation.candidate_sets
    validation_records = _planner_records(planner_validation)
    del planner_validation
    gc.collect()

    _write_json_atomic(
        root / "planner_supervision.json",
        {
            "training_candidate_sets": training_sets,
            "validation_candidate_sets": validation_sets,
            "candidates_per_set": 32,
            "no_op_index": 0,
            "progress_training_mean": progress_mean,
            "progress_training_std": progress_std,
            "training_records": training_records,
            "validation_records": validation_records,
        },
    )
    return {
        "transition_splits": transition_summary,
        "planner_training_candidate_sets": training_sets,
        "planner_validation_candidate_sets": validation_sets,
        "progress_training_mean": progress_mean,
        "progress_training_std": progress_std,
        "images_or_tensors_persisted": False,
    }


def run_manifest_compatibility_validation(
    repository_root: str | Path,
    output_dir: str | Path,
    config_path: str | Path = DEFAULT_COMPATIBILITY_CONFIG,
) -> dict[str, Any]:
    """Run one non-training compatibility attempt and preserve its four manifests."""

    repository = Path(repository_root).resolve()
    config = load_manifest_compatibility_config(repository / config_path)
    consumed = load_consumed_attempt_record(repository, config)
    base = load_phase_b_development_config(repository / str(config["base_config"]))
    if base.get("status") != ABORTED_STATUS:
        raise ValueError("Manifest compatibility requires the recovery-locked base phase.")
    if base["development"]["authorized"] is not False:
        raise PermissionError("Base Phase B0 development must remain unauthorized.")

    environment = runtime_environment_snapshot()
    validate_required_environment(config, environment)
    output = guard_output_directory(output_dir, repository)
    output.mkdir(parents=True)
    data_root = output / "data_manifests"
    started = time.perf_counter()
    generation = generate_manifest_only_data(base, data_root)
    elapsed = time.perf_counter() - started
    comparison = compare_manifest_hashes(
        data_root,
        config["expected_original_manifest_sha256"],
    )
    passed = bool(comparison["all_hashes_match"])
    report = {
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "experiment_id": config["experiment_id"],
        "compatibility_config_status": config["status"],
        "base_status": base["status"],
        "consumed_attempt_status": consumed["status"],
        "environment": environment,
        "generation": generation,
        "generation_seconds": elapsed,
        "manifest_comparison": comparison,
        "hash_gate_passed": passed,
        "execution_attempt_consumed": True,
        "old_authorization_reusable": False,
        "model_resources_loaded": False,
        "checkpoints_created": False,
        "scientific_models_trained": False,
        "recovery_output_created": False,
        "targets_or_images_persisted": False,
        "decision_created": False,
        "google_drive_accessed": False,
        "historical_incomplete_directories_touched": False,
        "historical_evidence_unchanged": True,
        "recovery_authorized": False,
        "formal_authorized": False,
        "phase_b1_authorized": False,
        "phase_b2_authorized": False,
        "passing_result_requires_separate_recovery_protocol": True,
        "passing_result_requires_separate_one_time_authorization": True,
    }
    _write_json_atomic(output / "manifest_compatibility_report.json", report)
    return report
