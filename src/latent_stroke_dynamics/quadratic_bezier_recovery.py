"""Fail-closed recovery for the interrupted fixed primitive comparison.

The recovery reuses byte-verified completed units, quarantines the one partial
unit without overwriting it, executes only missing units in the original frozen
schedule, and rebuilds the aggregate once all 36 units verify.  Validation is
side-effect free and recovery execution requires a separate authorization.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .quadratic_bezier_comparison import (
    EXPECTED_PAIR_COUNT,
    EXPECTED_RUN_COUNT,
    FREEZE_MANIFEST_PATH,
    FROZEN_TARGET_HASHES,
    FROZEN_TARGET_SET_SHA256,
    _ratio,
    _save_aggregate_progress_plot,
    _save_blinded_review_materials,
    _save_mean_plot,
    _save_target_ratio_plot,
    _write_pair_metrics,
    evaluate_quantitative_decision,
    run_target_seed_condition,
    validate_execution_authorization,
    validate_only_comparison_report,
    validate_target_freeze,
)
from .quadratic_bezier_extension import (
    PRIMITIVES,
    PROTOCOL_ID,
    SEEDS,
    TARGET_IDS,
    extension_painter_config,
    generate_rights_safe_targets,
    image_pixel_sha256,
)
from .rgb_coarse_to_fine import _write_json, file_sha256


FROZEN_RUNNER_COMMIT = "398a2bfb7bd65ed8b4bbc93fb8cc05564f7f3c1b"
RECOVERY_PLAN_PATH = Path(
    "configs/quadratic-bezier-interrupted-recovery-plan-2026-09-05.json"
)
RECOVERY_PLAN_STATUS = (
    "stable_interrupted_state_recovery_implementation_unauthorized"
)
RECOVERY_AUTHORIZATION_STATUS = (
    "authorized_for_single_interrupted_comparison_recovery"
)
ORIGINAL_AUTHORIZATION_ID = "quadratic_bezier_fixed_comparison_2026_09_04_v1"
RECOVERY_OUTPUT_NAMES = (
    "quantitative_decision.json",
    "target_seed_pair_metrics.csv",
    "mean_512_mse_by_primitive.png",
    "per_target_curve_ratio.png",
    "aggregate_progress_by_primitive.png",
    "blinded_review_montage.png",
    "blinded_mapping_do_not_open_before_review.json",
    "blinded_review_sheet.csv",
    "aggregate_summary.json",
    "aggregate_summary.sha256",
)
FROZEN_SOURCE_PATHS = (
    "pyproject.toml",
    "configs/quadratic-bezier-target-freeze-2026-09-04.json",
    "run_quadratic_bezier_comparison.py",
    "src/latent_stroke_dynamics/quadratic_bezier_comparison.py",
    "src/latent_stroke_dynamics/quadratic_bezier_extension.py",
    "src/latent_stroke_dynamics/quadratic_bezier_replay.py",
    "src/latent_stroke_dynamics/rgb_coarse_to_fine.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary audit file: {temporary}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def expected_unit_schedule() -> tuple[str, ...]:
    return tuple(
        f"{target_id}/seed_{seed}/{primitive}"
        for target_id in TARGET_IDS
        for seed in SEEDS
        for primitive in PRIMITIVES
    )


def _parse_unit(unit: str) -> tuple[str, int, str]:
    parts = unit.split("/")
    if len(parts) != 3 or not parts[1].startswith("seed_"):
        raise ValueError(f"Invalid comparison unit path: {unit}")
    target_id = parts[0]
    try:
        seed = int(parts[1][len("seed_") :])
    except ValueError as error:
        raise ValueError(f"Invalid comparison unit seed: {unit}") from error
    primitive = parts[2]
    if target_id not in TARGET_IDS or seed not in SEEDS or primitive not in PRIMITIVES:
        raise ValueError(f"Unit is outside the frozen schedule: {unit}")
    if unit not in expected_unit_schedule():
        raise ValueError(f"Unit is outside the frozen schedule: {unit}")
    return target_id, seed, primitive


def _strict_hash_tree(
    root: Path,
    *,
    excluded: set[str] | None = None,
) -> dict[str, str]:
    excluded = excluded or set()
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Symbolic links are prohibited in recovery state: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in excluded:
            result[relative] = file_sha256(path)
    return result


def _manifest_sha256(manifest: Mapping[str, str]) -> str:
    payload = json.dumps(
        dict(manifest), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def load_recovery_plan(path: Path = RECOVERY_PLAN_PATH) -> dict[str, Any]:
    plan = _read_json(path)
    exact = {
        "protocol_id": PROTOCOL_ID,
        "recovery_plan_id": "quadratic_bezier_interrupted_recovery_v1",
        "status": RECOVERY_PLAN_STATUS,
        "branch": "quadratic-bezier-extension",
        "frozen_runner_commit": FROZEN_RUNNER_COMMIT,
        "original_authorization_id": ORIGINAL_AUTHORIZATION_ID,
        "expected_output_name": "quadratic-bezier-fixed-comparison-v1",
        "expected_incomplete_name": "quadratic-bezier-fixed-comparison-v1.incomplete",
        "stable_diagnostic_repetitions": 2,
        "stable_diagnostic_captured_after_original_runtime_terminated": True,
        "expected_run_count": EXPECTED_RUN_COUNT,
        "expected_completed_run_count": 17,
        "expected_partial_run_count": 1,
        "expected_not_started_run_count": 18,
        "expected_missing_run_count": 19,
        "expected_file_count_before_recovery": 215,
        "expected_total_bytes_before_recovery": 6958862,
        "completed_units_all_integrity_checks_passed": True,
        "completed_units_must_be_preserved_byte_for_byte": True,
        "partial_unit_action": "hash_then_atomic_quarantine_without_overwrite",
        "missing_unit_execution_order": "original_frozen_schedule_only",
        "aggregate_rebuilt_only_after_36_verified_runs": True,
        "blinded_review_boundary_preserved": True,
        "fresh_comparison_execution_allowed": False,
        "automatic_resume_authorized": False,
        "recovery_execution_authorized": False,
        "maximum_completed_recoveries": 1,
        "completed_recoveries": 0,
        "metrics_revealed": False,
        "images_opened": False,
        "training_allowed": False,
        "learned_model_allowed": False,
        "comparative_tuning_allowed": False,
    }
    for key, expected in exact.items():
        if plan.get(key) != expected:
            raise ValueError(f"Frozen recovery plan mismatch for {key}.")

    schedule = expected_unit_schedule()
    completed = tuple(str(item) for item in plan.get("expected_completed_units", []))
    partial = tuple(str(item) for item in plan.get("expected_partial_units", []))
    missing = tuple(str(item) for item in plan.get("expected_missing_units", []))
    for unit in completed + partial + missing:
        _parse_unit(unit)
    if len(completed) != 17 or len(set(completed)) != 17:
        raise ValueError("Recovery plan completed-unit list is invalid.")
    if partial != ("03_organic_silhouette/seed_211/quadratic_bezier",):
        raise ValueError("Recovery plan partial-unit identity changed.")
    expected_missing = tuple(unit for unit in schedule if unit not in set(completed))
    if missing != expected_missing:
        raise ValueError("Recovery plan missing-unit order changed.")
    if set(completed) & set(partial):
        raise ValueError("A recovery unit cannot be complete and partial.")
    not_started = tuple(unit for unit in missing if unit not in set(partial))
    if len(not_started) != 18:
        raise ValueError("Recovery plan not-started count changed.")
    return plan


def validate_recovery_source_continuity(
    repo_root: Path,
    *,
    expected_head: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    observed_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if expected_head is not None and observed_head != expected_head:
        raise RuntimeError("Checked-out recovery implementation commit changed.")
    for relative in FROZEN_SOURCE_PATHS:
        comparison = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                f"{FROZEN_RUNNER_COMMIT}..HEAD",
                "--",
                relative,
            ],
            cwd=repo_root,
        )
        if comparison.returncode != 0:
            raise RuntimeError(f"Frozen runner dependency changed: {relative}")
    return {
        "frozen_runner_commit": FROZEN_RUNNER_COMMIT,
        "observed_recovery_commit": observed_head,
        "verified_unchanged_path_count": len(FROZEN_SOURCE_PATHS),
    }


def validate_completed_run_directory(
    run_dir: Path,
    *,
    unit: str,
) -> dict[str, Any]:
    target_id, seed, primitive = _parse_unit(unit)
    if not run_dir.is_dir():
        raise RuntimeError(f"Completed unit directory is missing: {unit}")
    summary_path = run_dir / "summary.json"
    summary_sha_path = run_dir / "summary.sha256"
    if not summary_path.is_file() or not summary_sha_path.is_file():
        raise RuntimeError(f"Completed unit evidence is incomplete: {unit}")
    recorded_summary_sha = summary_sha_path.read_text(encoding="utf-8").strip()
    if recorded_summary_sha != file_sha256(summary_path):
        raise RuntimeError(f"Completed unit summary checksum failed: {unit}")
    summary = _read_json(summary_path)
    exact = {
        "status": "quadratic_bezier_comparison_run_complete",
        "protocol_id": PROTOCOL_ID,
        "target_id": target_id,
        "target_stream": TARGET_IDS.index(target_id),
        "target_pixel_sha256": FROZEN_TARGET_HASHES[target_id],
        "primitive": primitive,
        "seed": seed,
        "planning_size": 128,
        "replay_size": 512,
        "maximum_strokes": 420,
        "candidates_per_pool": 64,
        "every_executed_stroke_improved": True,
        "best_not_worse_than_final": True,
        "training_performed": False,
        "learned_model_used": False,
        "closed_experiments_changed": False,
    }
    for key, expected in exact.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"Completed unit field failed for {unit}: {key}")

    numeric_fields = (
        "initial_mse",
        "best_mse",
        "final_mse",
        "final_mae",
        "high_resolution_best_mse",
        "high_resolution_final_mse",
        "high_resolution_final_mae",
        "runtime_seconds",
    )
    if any(
        not isinstance(summary.get(key), (int, float))
        or not isfinite(float(summary[key]))
        for key in numeric_fields
    ):
        raise RuntimeError(f"Completed unit has a non-finite value: {unit}")
    executed_strokes = summary.get("executed_strokes")
    candidate_pools = summary.get("candidate_pools")
    candidate_renders = summary.get("candidate_renders")
    if not isinstance(executed_strokes, int) or not 1 <= executed_strokes <= 420:
        raise RuntimeError(f"Completed unit stroke count failed: {unit}")
    if not isinstance(candidate_pools, int) or candidate_pools < executed_strokes:
        raise RuntimeError(f"Completed unit candidate-pool count failed: {unit}")
    if candidate_renders != candidate_pools * 64:
        raise RuntimeError(f"Completed unit candidate-render count failed: {unit}")
    if float(summary["best_mse"]) > float(summary["final_mse"]):
        raise RuntimeError(f"Completed unit best/final relation failed: {unit}")

    stage_stats = summary.get("stage_stats")
    if not isinstance(stage_stats, list) or len(stage_stats) != 3:
        raise RuntimeError(f"Completed unit stage evidence failed: {unit}")
    if [item.get("maximum_steps") for item in stage_stats] != [80, 140, 200]:
        raise RuntimeError(f"Completed unit stage budgets changed: {unit}")
    if sum(int(item.get("executed_steps", -1)) for item in stage_stats) != executed_strokes:
        raise RuntimeError(f"Completed unit executed-stage counts failed: {unit}")
    if sum(int(item.get("candidate_pools", -1)) for item in stage_stats) != candidate_pools:
        raise RuntimeError(f"Completed unit stage-pool counts failed: {unit}")

    artifact_manifest = summary.get("artifact_sha256")
    if not isinstance(artifact_manifest, dict) or not artifact_manifest:
        raise RuntimeError(f"Completed unit artifact manifest is missing: {unit}")
    if any(
        not isinstance(relative, str)
        or not isinstance(digest, str)
        or len(digest) != 64
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        for relative, digest in artifact_manifest.items()
    ):
        raise RuntimeError(f"Completed unit artifact manifest is unsafe: {unit}")
    observed_manifest = _strict_hash_tree(
        run_dir,
        excluded={"summary.json", "summary.sha256"},
    )
    if observed_manifest != artifact_manifest:
        raise RuntimeError(f"Completed unit artifact hashes failed: {unit}")

    run_config = _read_json(run_dir / "run_config.json")
    expected_config = json.loads(json.dumps(asdict(extension_painter_config(seed))))
    run_config_exact = {
        "protocol_id": PROTOCOL_ID,
        "target_id": target_id,
        "target_stream": TARGET_IDS.index(target_id),
        "target_pixel_sha256": FROZEN_TARGET_HASHES[target_id],
        "primitive": primitive,
        "seed": seed,
        "painter_config": expected_config,
        "selection": "exact_rendered_rgb_target_pixel_mse",
        "training_performed": False,
        "learned_model_used": False,
    }
    if run_config != run_config_exact:
        raise RuntimeError(f"Completed unit run configuration changed: {unit}")

    actions = json.loads((run_dir / "actions.json").read_text(encoding="utf-8"))
    if not isinstance(actions, list) or len(actions) != executed_strokes:
        raise RuntimeError(f"Completed unit action count failed: {unit}")
    if any(item.get("primitive") != primitive for item in actions):
        raise RuntimeError(f"Completed unit action primitive failed: {unit}")
    with (run_dir / "progress.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        progress = list(csv.DictReader(handle))
    if len(progress) != executed_strokes:
        raise RuntimeError(f"Completed unit progress length failed: {unit}")
    for index, row in enumerate(progress, start=1):
        before = float(row["mse_before"])
        after = float(row["mse_after"])
        improvement = float(row["improvement"])
        if int(row["step"]) != index or row["primitive"] != primitive:
            raise RuntimeError(f"Completed unit progress identity failed: {unit}")
        if not all(isfinite(value) for value in (before, after, improvement)):
            raise RuntimeError(f"Completed unit progress is non-finite: {unit}")
        if not after < before or not improvement > 1e-9:
            raise RuntimeError(f"Completed unit progress monotonicity failed: {unit}")
    return summary


def inspect_stable_interrupted_state(
    *,
    output_dir: Path,
    plan: Mapping[str, Any],
    freeze: Mapping[str, Any],
    original_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists():
        raise FileExistsError(f"Completed output already exists: {output_dir}")
    if not incomplete.is_dir():
        raise FileNotFoundError(f"Preserved incomplete output is missing: {incomplete}")
    if incomplete.name != plan["expected_incomplete_name"]:
        raise RuntimeError("Preserved incomplete output name changed.")
    for name in RECOVERY_OUTPUT_NAMES + ("failure.json",):
        if (incomplete / name).exists():
            raise RuntimeError(f"Unexpected pre-recovery aggregate artifact exists: {name}")
    if (incomplete / "recovery_audit").exists() or (
        incomplete / "interrupted_attempt_quarantine"
    ).exists():
        raise RuntimeError("A recovery mutation already exists; stop for a new audit.")

    files = [path for path in incomplete.rglob("*") if path.is_file()]
    if len(files) != int(plan["expected_file_count_before_recovery"]):
        raise RuntimeError("Stable diagnostic file count changed.")
    if sum(path.stat().st_size for path in files) != int(
        plan["expected_total_bytes_before_recovery"]
    ):
        raise RuntimeError("Stable diagnostic byte count changed.")

    if _read_json(incomplete / "target_freeze_manifest.json") != dict(freeze):
        raise RuntimeError("Saved target-freeze manifest changed.")
    if _read_json(incomplete / "execution_authorization.json") != dict(
        original_authorization
    ):
        raise RuntimeError("Saved original execution authorization changed.")
    environment = _read_json(incomplete / "environment_manifest.json")
    if environment.get("source_commit") != FROZEN_RUNNER_COMMIT:
        raise RuntimeError("Saved interrupted-run source commit changed.")
    if environment.get("target_freeze_sha256") != file_sha256(FREEZE_MANIFEST_PATH):
        raise RuntimeError("Saved interrupted-run target-freeze hash changed.")

    runs_root = incomplete / "runs"
    if not runs_root.is_dir():
        raise RuntimeError("Interrupted run tree is missing.")
    discovered_dirs = {
        path.relative_to(runs_root).as_posix(): path
        for path in runs_root.glob("*/*/*")
        if path.is_dir()
    }
    schedule = set(expected_unit_schedule())
    unexpected = set(discovered_dirs) - schedule
    if unexpected:
        raise RuntimeError("Unexpected run directories exist in interrupted state.")
    completed = {
        unit for unit, path in discovered_dirs.items() if (path / "summary.json").is_file()
    }
    partial = set(discovered_dirs) - completed
    expected_completed = set(str(item) for item in plan["expected_completed_units"])
    expected_partial = set(str(item) for item in plan["expected_partial_units"])
    if completed != expected_completed:
        raise RuntimeError("Stable diagnostic completed-unit set changed.")
    if partial != expected_partial:
        raise RuntimeError("Stable diagnostic partial-unit set changed.")
    not_started = schedule - set(discovered_dirs)
    if len(not_started) != int(plan["expected_not_started_run_count"]):
        raise RuntimeError("Stable diagnostic not-started set changed.")

    completed_manifests: dict[str, dict[str, str]] = {}
    completed_manifest_sha256: dict[str, str] = {}
    for unit in plan["expected_completed_units"]:
        run_dir = runs_root / str(unit)
        validate_completed_run_directory(run_dir, unit=str(unit))
        manifest = _strict_hash_tree(run_dir)
        completed_manifests[str(unit)] = manifest
        completed_manifest_sha256[str(unit)] = _manifest_sha256(manifest)
    partial_unit = str(plan["expected_partial_units"][0])
    partial_manifest = _strict_hash_tree(runs_root / partial_unit)
    return {
        "completed_run_count": len(completed),
        "partial_run_count": len(partial),
        "not_started_run_count": len(not_started),
        "missing_run_count": len(expected_unit_schedule()) - len(completed),
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "completed_manifests": completed_manifests,
        "completed_manifest_sha256": completed_manifest_sha256,
        "partial_unit": partial_unit,
        "partial_manifest": partial_manifest,
        "partial_manifest_sha256": _manifest_sha256(partial_manifest),
    }


def validate_recovery_authorization(
    path: Path,
    *,
    recovery_implementation_commit: str,
    plan_path: Path = RECOVERY_PLAN_PATH,
) -> dict[str, Any]:
    plan = load_recovery_plan(plan_path)
    authorization = _read_json(path)
    exact = {
        "protocol_id": PROTOCOL_ID,
        "recovery_plan_id": plan["recovery_plan_id"],
        "status": RECOVERY_AUTHORIZATION_STATUS,
        "branch": "quadratic-bezier-extension",
        "frozen_runner_commit": FROZEN_RUNNER_COMMIT,
        "recovery_implementation_commit": recovery_implementation_commit,
        "recovery_plan_sha256": file_sha256(plan_path),
        "original_execution_authorization_id": ORIGINAL_AUTHORIZATION_ID,
        "expected_output_name": plan["expected_output_name"],
        "expected_incomplete_name": plan["expected_incomplete_name"],
        "expected_completed_run_count": 17,
        "expected_partial_run_count": 1,
        "expected_missing_run_count": 19,
        "expected_partial_units": plan["expected_partial_units"],
        "allowed_entry_point": "run_quadratic_bezier_recovery.py --execute-recovery",
        "maximum_completed_recoveries": 1,
        "completed_recoveries": 0,
        "recovery_execution_authorized": True,
        "fresh_comparison_execution_allowed": False,
        "overwrite_completed_units_allowed": False,
        "overwrite_partial_unit_allowed": False,
        "quarantine_partial_unit_required": True,
        "comparative_tuning_allowed": False,
        "metrics_may_be_revealed_before_gate": False,
        "training_allowed": False,
        "learned_model_allowed": False,
    }
    for key, expected in exact.items():
        if authorization.get(key) != expected:
            raise PermissionError(f"Recovery authorization mismatch for {key}.")
    return authorization


def validate_only_recovery_report(
    *,
    repo_root: Path,
    plan_path: Path = RECOVERY_PLAN_PATH,
    freeze_path: Path = FREEZE_MANIFEST_PATH,
    expected_head: str | None = None,
) -> dict[str, Any]:
    plan = load_recovery_plan(plan_path)
    continuity = validate_recovery_source_continuity(
        repo_root, expected_head=expected_head
    )
    runner = validate_only_comparison_report(freeze_path)
    return {
        "status": "quadratic_bezier_recovery_implementation_valid_no_outputs_unauthorized",
        "protocol_id": PROTOCOL_ID,
        "recovery_plan_sha256": file_sha256(plan_path),
        "frozen_runner_commit": FROZEN_RUNNER_COMMIT,
        "observed_recovery_commit": continuity["observed_recovery_commit"],
        "verified_unchanged_source_path_count": continuity[
            "verified_unchanged_path_count"
        ],
        "runner_validation_status": runner["status"],
        "target_freeze_sha256": runner["target_freeze_sha256"],
        "target_set_sha256": runner["target_set_sha256"],
        "expected_run_count": EXPECTED_RUN_COUNT,
        "preserved_completed_run_count": plan["expected_completed_run_count"],
        "quarantined_partial_run_count": plan["expected_partial_run_count"],
        "missing_run_count": plan["expected_missing_run_count"],
        "recovery_execution_authorized": False,
        "interrupted_output_accessed": False,
        "interrupted_output_modified": False,
        "recovery_output_created": False,
        "comparative_metrics_revealed": False,
        "images_opened": False,
        "training_performed": False,
        "learned_model_used": False,
        "closed_experiments_changed": False,
        "environment": runner["environment"],
    }


def _append_recovery_event(
    journal_path: Path,
    stage: str,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    if journal_path.exists():
        journal = _read_json(journal_path)
    else:
        journal = {
            "status": "quadratic_bezier_recovery_in_progress",
            "automatic_resume_authorized": False,
            "metrics_recorded_in_journal": False,
            "events": [],
        }
    events = journal.get("events")
    if not isinstance(events, list):
        raise RuntimeError("Recovery journal is invalid.")
    event: dict[str, Any] = {
        "stage": stage,
        "status": status,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if details:
        event["details"] = dict(details)
    events.append(event)
    _write_json_atomic(journal_path, journal)


def _quarantine_partial_unit(
    incomplete: Path,
    *,
    unit: str,
    expected_manifest: Mapping[str, str],
) -> Path:
    source = incomplete / "runs" / unit
    destination = incomplete / "interrupted_attempt_quarantine" / "runs" / unit
    if not source.is_dir():
        raise RuntimeError("The expected partial unit is missing before quarantine.")
    if destination.exists():
        raise FileExistsError("Partial-unit quarantine destination already exists.")
    observed = _strict_hash_tree(source)
    if observed != dict(expected_manifest):
        raise RuntimeError("Partial-unit bytes changed before quarantine.")
    destination.parent.mkdir(parents=True, exist_ok=False)
    source.rename(destination)
    if _strict_hash_tree(destination) != dict(expected_manifest):
        raise RuntimeError("Partial-unit bytes changed during quarantine.")
    return destination


def _verify_preserved_completed_units(
    incomplete: Path,
    manifests: Mapping[str, Mapping[str, str]],
) -> None:
    for unit, expected in manifests.items():
        if _strict_hash_tree(incomplete / "runs" / unit) != dict(expected):
            raise RuntimeError(f"A preserved completed unit changed: {unit}")


def _load_verified_summaries(
    incomplete: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for target_id in TARGET_IDS:
        for seed in SEEDS:
            pair: dict[str, dict[str, Any]] = {}
            for primitive in PRIMITIVES:
                unit = f"{target_id}/seed_{seed}/{primitive}"
                summary = validate_completed_run_directory(
                    incomplete / "runs" / unit,
                    unit=unit,
                )
                pair[primitive] = summary
                summaries.append(summary)
            straight = pair["straight"]
            curve = pair["quadratic_bezier"]
            straight_mse = float(straight["high_resolution_final_mse"])
            curve_mse = float(curve["high_resolution_final_mse"])
            ratio = _ratio(curve_mse, straight_mse)
            pair_rows.append(
                {
                    "target_id": target_id,
                    "seed": seed,
                    "straight_mse_512": straight_mse,
                    "quadratic_bezier_mse_512": curve_mse,
                    "curve_to_straight_ratio": ratio,
                    "curve_relative_change": ratio - 1.0,
                    "straight_candidate_renders": straight["candidate_renders"],
                    "quadratic_bezier_candidate_renders": curve[
                        "candidate_renders"
                    ],
                    "straight_runtime_seconds": straight["runtime_seconds"],
                    "quadratic_bezier_runtime_seconds": curve["runtime_seconds"],
                }
            )
    return summaries, pair_rows


def _finalize_recovered_aggregate(
    *,
    incomplete: Path,
    output_dir: Path,
    original_authorization: Mapping[str, Any],
    recovery_authorization: Mapping[str, Any],
    recovery_implementation_commit: str,
    freeze_path: Path,
    recovery_started: float,
    journal_path: Path,
    preserved_manifests: Mapping[str, Mapping[str, str]],
    quarantined_manifest: Mapping[str, str],
    partial_unit: str,
) -> dict[str, Any]:
    for name in RECOVERY_OUTPUT_NAMES:
        if (incomplete / name).exists():
            raise RuntimeError(f"Refusing to overwrite aggregate artifact: {name}")
    _verify_preserved_completed_units(incomplete, preserved_manifests)
    quarantine = incomplete / "interrupted_attempt_quarantine" / "runs" / partial_unit
    if _strict_hash_tree(quarantine) != dict(quarantined_manifest):
        raise RuntimeError("Quarantined partial-unit bytes changed.")

    summaries, pair_rows = _load_verified_summaries(incomplete)
    targets = generate_rights_safe_targets(512)
    if any(
        image_pixel_sha256(target.image) != FROZEN_TARGET_HASHES[target.target_id]
        for target in targets
    ):
        raise RuntimeError("Frozen procedural target changed before aggregation.")
    integrity = {
        "all_expected_runs_completed": len(summaries) == EXPECTED_RUN_COUNT,
        "all_expected_pairs_completed": len(pair_rows) == EXPECTED_PAIR_COUNT,
        "all_run_artifact_manifests_verified": len(summaries) == EXPECTED_RUN_COUNT,
        "all_pre_interruption_completed_units_byte_preserved": True,
        "interrupted_partial_unit_quarantined_byte_preserved": True,
        "all_executed_strokes_improved": all(
            bool(item["every_executed_stroke_improved"]) for item in summaries
        ),
        "all_best_frames_not_worse_than_final": all(
            bool(item["best_not_worse_than_final"]) for item in summaries
        ),
        "all_target_hashes_preserved": all(
            item["target_pixel_sha256"]
            == FROZEN_TARGET_HASHES[str(item["target_id"])]
            for item in summaries
        ),
        "no_training_performed": all(
            item["training_performed"] is False for item in summaries
        ),
        "no_learned_model_used": all(
            item["learned_model_used"] is False for item in summaries
        ),
        "closed_experiments_unchanged": all(
            item["closed_experiments_changed"] is False for item in summaries
        ),
    }
    integrity_passed = all(integrity.values())
    decision = evaluate_quantitative_decision(
        pair_rows,
        integrity_passed=integrity_passed,
    )
    _write_json(incomplete / "quantitative_decision.json", decision)
    _write_pair_metrics(incomplete / "target_seed_pair_metrics.csv", pair_rows)
    _save_mean_plot(incomplete / "mean_512_mse_by_primitive.png", decision)
    _save_target_ratio_plot(incomplete / "per_target_curve_ratio.png", decision)
    _save_aggregate_progress_plot(
        incomplete / "aggregate_progress_by_primitive.png",
        incomplete / "runs",
    )
    blinded = _save_blinded_review_materials(incomplete, targets=targets)

    total_candidate_renders = {
        primitive: sum(
            int(item["candidate_renders"])
            for item in summaries
            if item["primitive"] == primitive
        )
        for primitive in PRIMITIVES
    }
    total_runtime_seconds = {
        primitive: fsum(
            float(item["runtime_seconds"])
            for item in summaries
            if item["primitive"] == primitive
        )
        for primitive in PRIMITIVES
    }
    _append_recovery_event(
        journal_path,
        "aggregate_ready_for_atomic_finalize",
        "completed",
        {
            "verified_run_count": len(summaries),
            "verified_pair_count": len(pair_rows),
            "integrity_passed": integrity_passed,
            "metrics_withheld": True,
        },
    )
    aggregate: dict[str, Any] = {
        "status": "quadratic_bezier_fixed_comparison_complete",
        "completion_mode": "verified_interrupted_attempt_recovery",
        "protocol_id": PROTOCOL_ID,
        "source_commit": FROZEN_RUNNER_COMMIT,
        "recovery_implementation_commit": recovery_implementation_commit,
        "authorization_id": original_authorization.get("authorization_id"),
        "recovery_authorization_id": recovery_authorization.get("authorization_id"),
        "target_set_sha256": FROZEN_TARGET_SET_SHA256,
        "target_freeze_sha256": file_sha256(freeze_path),
        "completed_run_count": len(summaries),
        "completed_pair_count": len(pair_rows),
        "reused_completed_run_count": len(preserved_manifests),
        "executed_during_recovery_run_count": EXPECTED_RUN_COUNT
        - len(preserved_manifests),
        "quarantined_partial_unit": partial_unit,
        "run_summaries": summaries,
        "pair_metrics": pair_rows,
        "integrity": integrity,
        "integrity_passed": integrity_passed,
        "quantitative_decision": decision,
        "blinded_review": blinded,
        "total_candidate_renders": total_candidate_renders,
        "condition_runtime_seconds": total_runtime_seconds,
        "recovery_wall_clock_seconds": time.perf_counter() - recovery_started,
        "wall_clock_scope": (
            "Recovery interval only; exact per-condition totals include all 36 "
            "verified run runtimes. The interrupted-attempt wall clock is documented "
            "separately and is not reconstructed."
        ),
        "training_performed": False,
        "learned_model_used": False,
        "closed_experiments_changed": False,
        "final_decision_requires_blinded_review": bool(
            decision["qualitative_review_required"]
        ),
        "output_directory": str(output_dir),
    }
    aggregate["artifact_sha256"] = _strict_hash_tree(
        incomplete,
        excluded={
            "aggregate_summary.json",
            "aggregate_summary.sha256",
            "failure.json",
            "recovery_audit/recovery_failure.json",
        },
    )
    _write_json(incomplete / "aggregate_summary.json", aggregate)
    (incomplete / "aggregate_summary.sha256").write_text(
        file_sha256(incomplete / "aggregate_summary.json") + "\n",
        encoding="utf-8",
    )
    if output_dir.exists():
        raise FileExistsError("Completed output appeared before atomic finalize.")
    incomplete.rename(output_dir)
    return aggregate


def run_interrupted_comparison_recovery(
    *,
    output_dir: Path,
    original_authorization_path: Path,
    recovery_authorization_path: Path,
    recovery_implementation_commit: str,
    repo_root: Path,
    plan_path: Path = RECOVERY_PLAN_PATH,
    freeze_path: Path = FREEZE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Complete one audited interrupted attempt without rerunning valid units."""

    plan = load_recovery_plan(plan_path)
    if output_dir.name != plan["expected_output_name"]:
        raise PermissionError("Recovery output name does not match the frozen plan.")
    validate_recovery_source_continuity(
        repo_root,
        expected_head=recovery_implementation_commit,
    )
    freeze = validate_target_freeze(freeze_path)
    original_authorization = validate_execution_authorization(
        original_authorization_path,
        source_commit=FROZEN_RUNNER_COMMIT,
        freeze_path=freeze_path,
    )
    recovery_authorization = validate_recovery_authorization(
        recovery_authorization_path,
        recovery_implementation_commit=recovery_implementation_commit,
        plan_path=plan_path,
    )
    initial = inspect_stable_interrupted_state(
        output_dir=output_dir,
        plan=plan,
        freeze=freeze,
        original_authorization=original_authorization,
    )

    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    audit_root = incomplete / "recovery_audit"
    journal_path = audit_root / "recovery_journal.json"
    recovery_started = time.perf_counter()
    mutation_started = False
    try:
        _write_json_atomic(
            audit_root / "initial_state_manifest.json",
            {
                "status": "stable_interrupted_state_verified_before_recovery",
                "protocol_id": PROTOCOL_ID,
                "frozen_runner_commit": FROZEN_RUNNER_COMMIT,
                "recovery_implementation_commit": recovery_implementation_commit,
                "recovery_plan_sha256": file_sha256(plan_path),
                "original_authorization_sha256": file_sha256(
                    original_authorization_path
                ),
                "recovery_authorization_sha256": file_sha256(
                    recovery_authorization_path
                ),
                "completed_run_count": initial["completed_run_count"],
                "partial_run_count": initial["partial_run_count"],
                "not_started_run_count": initial["not_started_run_count"],
                "missing_run_count": initial["missing_run_count"],
                "pre_recovery_file_count": initial["file_count"],
                "pre_recovery_total_bytes": initial["total_bytes"],
                "completed_unit_file_manifests": initial["completed_manifests"],
                "completed_unit_manifest_sha256": initial[
                    "completed_manifest_sha256"
                ],
                "partial_unit": initial["partial_unit"],
                "partial_unit_file_manifest": initial["partial_manifest"],
                "partial_unit_manifest_sha256": initial[
                    "partial_manifest_sha256"
                ],
                "metrics_included": False,
                "images_opened": False,
            },
        )
        mutation_started = True
        _append_recovery_event(
            journal_path,
            "stable_state_verified",
            "completed",
            {
                "preserved_completed_run_count": initial["completed_run_count"],
                "partial_run_count": initial["partial_run_count"],
                "missing_run_count": initial["missing_run_count"],
            },
        )
        _quarantine_partial_unit(
            incomplete,
            unit=initial["partial_unit"],
            expected_manifest=initial["partial_manifest"],
        )
        _append_recovery_event(
            journal_path,
            "partial_unit_quarantined",
            "completed",
            {
                "unit": initial["partial_unit"],
                "manifest_sha256": initial["partial_manifest_sha256"],
            },
        )

        targets = generate_rights_safe_targets(512)
        target_by_id = {target.target_id: target for target in targets}
        for target in targets:
            if image_pixel_sha256(target.image) != FROZEN_TARGET_HASHES[target.target_id]:
                raise RuntimeError(f"Frozen target changed: {target.target_id}")
        for unit in plan["expected_missing_units"]:
            target_id, seed, primitive = _parse_unit(str(unit))
            run_dir = incomplete / "runs" / str(unit)
            if run_dir.exists():
                raise FileExistsError(f"Refusing to overwrite a recovery unit: {unit}")
            summary = run_target_seed_condition(
                target_by_id[target_id],
                primitive=primitive,
                seed=seed,
                target_stream=TARGET_IDS.index(target_id),
                output_dir=run_dir,
            )
            validate_completed_run_directory(run_dir, unit=str(unit))
            _verify_preserved_completed_units(
                incomplete, initial["completed_manifests"]
            )
            _append_recovery_event(
                journal_path,
                "missing_unit_completed",
                "completed",
                {
                    "unit": str(unit),
                    "summary_sha256": file_sha256(run_dir / "summary.json"),
                    "artifact_count": len(summary["artifact_sha256"]),
                    "metrics_withheld": True,
                },
            )

        return _finalize_recovered_aggregate(
            incomplete=incomplete,
            output_dir=output_dir,
            original_authorization=original_authorization,
            recovery_authorization=recovery_authorization,
            recovery_implementation_commit=recovery_implementation_commit,
            freeze_path=freeze_path,
            recovery_started=recovery_started,
            journal_path=journal_path,
            preserved_manifests=initial["completed_manifests"],
            quarantined_manifest=initial["partial_manifest"],
            partial_unit=initial["partial_unit"],
        )
    except BaseException as error:
        if mutation_started:
            failure_root = incomplete if incomplete.exists() else output_dir
            try:
                _write_json_atomic(
                    failure_root / "recovery_audit" / "recovery_failure.json",
                    {
                        "status": "quadratic_bezier_recovery_interrupted_or_failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "automatic_resume_authorized": False,
                        "fresh_execution_authorized": False,
                        "metrics_revealed": False,
                        "training_performed": False,
                        "learned_model_used": False,
                        "closed_experiments_changed": False,
                    },
                )
            except Exception:
                pass
        raise
