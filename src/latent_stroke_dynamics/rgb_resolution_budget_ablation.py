"""Guarded 2x2 planning-resolution and stroke-budget RGB ablation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from math import fsum
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .rgb_coarse_to_fine import (
    DEFAULT_STAGES,
    PainterConfig,
    StageConfig,
    TARGET_SET_SHA256,
    TARGET_SPECS,
    _hash_tree,
    _make_montage,
    _save_aggregate_progress_plot,
    _write_json,
    file_sha256,
    plan_rgb_target,
    run_one_target,
    validate_fixed_targets,
)


PROTOCOL_ID = "rgb_resolution_budget_ablation_v1"
BASELINE_COMMIT = "2c50e88e4c499f30433eb6884b51d147fe23bfa5"
BASELINE_ARCHIVE_COMMIT = "3aad9332a11a101a44656b93434c1adff6c83000"
BASELINE_AGGREGATE_SHA256 = (
    "7f2c32cef077bef1737a3f00ee584cf1075b1feb5711995ab87dd9812f233c05"
)
BASELINE_MEAN_512_MSE = 0.012319037602067864
MEAN_IMPROVEMENT_RATIO = 0.90
MAX_TARGET_WORSENING_RATIO = 1.05
BEST_MEAN_TOLERANCE_RATIO = 1.01


@dataclass(frozen=True)
class AblationCondition:
    condition_id: str
    slug: str
    label: str
    planning_size: int
    budget_multiplier: int
    execute: bool

    @property
    def total_strokes(self) -> int:
        return self.budget_multiplier * sum(stage.max_steps for stage in DEFAULT_STAGES)

    @property
    def compute_proxy(self) -> int:
        return self.planning_size * self.planning_size * self.total_strokes


CONDITIONS = (
    AblationCondition("A", "A_baseline_96x210", "archived baseline", 96, 1, False),
    AblationCondition("B", "B_budget_96x420", "budget only", 96, 2, True),
    AblationCondition("C", "C_resolution_128x210", "resolution only", 128, 1, True),
    AblationCondition("D", "D_resolution_budget_128x420", "resolution + budget", 128, 2, True),
)
NEW_CONDITIONS = tuple(condition for condition in CONDITIONS if condition.execute)


def _scaled_stages(multiplier: int) -> tuple[StageConfig, ...]:
    if multiplier < 1:
        raise ValueError("Stage-budget multiplier must be positive.")
    return tuple(
        StageConfig(
            stage.name,
            stage.max_steps * multiplier,
            stage.min_length,
            stage.max_length,
            stage.min_width,
            stage.max_width,
        )
        for stage in DEFAULT_STAGES
    )


def config_for_condition(condition: AblationCondition) -> PainterConfig:
    config = PainterConfig(
        planning_size=condition.planning_size,
        replay_size=512,
        supersample=2,
        candidates_per_pool=64,
        error_guided_fraction=0.80,
        patience=12,
        min_improvement=1e-9,
        seed=73,
        gif_stride=3,
        max_attempts_per_candidate=400,
        stages=_scaled_stages(condition.budget_multiplier),
    )
    config.validate()
    return config


def protocol_manifest() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "schema_version": 1,
        "target_set_sha256": TARGET_SET_SHA256,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_archive_commit": BASELINE_ARCHIVE_COMMIT,
        "baseline_aggregate_sha256": BASELINE_AGGREGATE_SHA256,
        "baseline_mean_512_mse": BASELINE_MEAN_512_MSE,
        "conditions": [
            {
                **asdict(condition),
                "total_strokes": condition.total_strokes,
                "compute_proxy": condition.compute_proxy,
                "painter_config": asdict(config_for_condition(condition)),
            }
            for condition in CONDITIONS
        ],
        "decision_thresholds": {
            "required_mean_512_mse_ratio": MEAN_IMPROVEMENT_RATIO,
            "maximum_per_target_512_mse_ratio": MAX_TARGET_WORSENING_RATIO,
            "least_expensive_within_best_mean_ratio": BEST_MEAN_TOLERANCE_RATIO,
        },
        "primary_metric": "mean_common_target_512_rgb_mse",
        "training_performed": False,
        "learned_model_used": False,
        "frozen_phase_b0_decision_changed": False,
    }


def validate_protocol() -> dict[str, Any]:
    identifiers = [condition.condition_id for condition in CONDITIONS]
    if identifiers != ["A", "B", "C", "D"]:
        raise RuntimeError("Ablation condition order changed.")
    expected = {
        "A": (96, 210, False),
        "B": (96, 420, True),
        "C": (128, 210, True),
        "D": (128, 420, True),
    }
    for condition in CONDITIONS:
        actual = (condition.planning_size, condition.total_strokes, condition.execute)
        if actual != expected[condition.condition_id]:
            raise RuntimeError(f"Condition {condition.condition_id} changed: {actual}")
        config = config_for_condition(condition)
        if config.replay_size != 512 or config.candidates_per_pool != 64:
            raise RuntimeError("A fixed non-factor setting changed.")
        if config.seed != 73 or config.error_guided_fraction != 0.80:
            raise RuntimeError("A fixed stochastic setting changed.")
    return protocol_manifest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _target_map(summary: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    targets = summary.get("targets")
    if not isinstance(targets, list):
        raise ValueError("Aggregate summary has no target list.")
    mapped: dict[str, dict[str, Any]] = {}
    for item in targets:
        if not isinstance(item, dict) or not isinstance(item.get("target_id"), str):
            raise ValueError("Malformed target summary.")
        target_id = item["target_id"]
        if target_id in mapped:
            raise ValueError(f"Duplicate target summary: {target_id}")
        mapped[target_id] = item
    expected = [spec.target_id for spec in TARGET_SPECS]
    if list(mapped) != expected:
        raise ValueError(f"Unexpected target order: {list(mapped)}")
    return mapped


def validate_baseline_result(
    baseline_dir: Path,
    *,
    verify_artifacts: bool = True,
) -> tuple[dict[str, Any], int]:
    """Verify the immutable baseline summary and, by default, its artifact tree."""

    if not baseline_dir.is_dir():
        raise FileNotFoundError(f"Baseline directory not found: {baseline_dir}")
    summary_path = baseline_dir / "aggregate_summary.json"
    checksum_path = baseline_dir / "aggregate_summary.sha256"
    if not summary_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError("Baseline aggregate summary or checksum is missing.")

    observed_summary_hash = file_sha256(summary_path)
    if observed_summary_hash != BASELINE_AGGREGATE_SHA256:
        raise ValueError(
            "Baseline aggregate-summary SHA-256 mismatch; refusing substitution."
        )
    recorded_hash = checksum_path.read_text(encoding="utf-8").strip()
    if recorded_hash != observed_summary_hash:
        raise ValueError("Baseline aggregate checksum file does not match its summary.")

    summary = _load_json(summary_path)
    if summary.get("status") != "rgb_coarse_to_fine_fixed_set_complete":
        raise ValueError("Baseline completion status is not valid.")
    if summary.get("target_set_sha256") != TARGET_SET_SHA256:
        raise ValueError("Baseline target-set hash changed.")
    if summary.get("completed_target_count") != len(TARGET_SPECS):
        raise ValueError("Baseline does not contain exactly five completed targets.")
    if summary.get("training_performed") is not False:
        raise ValueError("Baseline unexpectedly records training.")
    if summary.get("learned_model_used") is not False:
        raise ValueError("Baseline unexpectedly records a learned model.")

    config = summary.get("painter_config")
    if not isinstance(config, dict):
        raise ValueError("Baseline painter configuration is missing.")
    if config.get("planning_size") != 96 or config.get("replay_size") != 512:
        raise ValueError("Baseline resolution changed.")
    if config.get("seed") != 73 or config.get("candidates_per_pool") != 64:
        raise ValueError("Baseline stochastic configuration changed.")
    stages = config.get("stages")
    if not isinstance(stages, list):
        raise ValueError("Baseline stage configuration is missing.")
    if [item.get("max_steps") for item in stages] != [40, 70, 100]:
        raise ValueError("Baseline stage budgets changed.")

    target_map = _target_map(summary)
    mean_mse = fsum(
        float(target_map[spec.target_id]["high_resolution_final_mse"])
        for spec in TARGET_SPECS
    ) / len(TARGET_SPECS)
    if not np.isclose(mean_mse, BASELINE_MEAN_512_MSE, rtol=0.0, atol=1e-15):
        raise ValueError("Baseline mean 512-pixel MSE changed.")

    artifact_hashes = summary.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("Baseline artifact manifest is missing.")
    verified = 0
    if verify_artifacts:
        for relative_path, expected_hash in sorted(artifact_hashes.items()):
            if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
                raise ValueError("Malformed baseline artifact manifest entry.")
            path = baseline_dir / relative_path
            if not path.is_file():
                raise FileNotFoundError(f"Baseline artifact missing: {relative_path}")
            if file_sha256(path) != expected_hash:
                raise ValueError(f"Baseline artifact hash mismatch: {relative_path}")
            verified += 1
    return summary, verified


def _run_condition(
    condition: AblationCondition,
    *,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not condition.execute:
        raise ValueError("The archived baseline must never be rerun.")
    config = config_for_condition(condition)
    output_dir.mkdir(parents=False, exist_ok=False)
    started = time.perf_counter()
    _write_json(
        output_dir / "condition_manifest.json",
        {
            "protocol_id": PROTOCOL_ID,
            "condition": asdict(condition),
            "target_set_sha256": TARGET_SET_SHA256,
            "painter_config": asdict(config),
            "selection": "exact_rendered_rgb_target_pixel_mse",
            "training_performed": False,
            "learned_model_used": False,
        },
    )

    summaries: list[dict[str, Any]] = []
    for target_stream, spec in enumerate(TARGET_SPECS):
        summaries.append(
            run_one_target(
                spec,
                input_dir=input_dir,
                output_dir=output_dir / spec.target_id,
                config=config,
                target_stream=target_stream,
            )
        )

    montage_items: list[tuple[str, Image.Image]] = []
    for spec in TARGET_SPECS:
        target_dir = output_dir / spec.target_id
        for suffix, filename in (
            ("target", "processed_target_512.png"),
            ("best", "best_512.png"),
            ("final", "final_512.png"),
        ):
            with Image.open(target_dir / filename) as image:
                montage_items.append(
                    (f"{spec.target_id} {suffix}", image.convert("RGB").copy())
                )
    _make_montage(montage_items, columns=3).save(
        output_dir / "five_target_montage.png"
    )
    _save_aggregate_progress_plot(output_dir, summaries)

    aggregate: dict[str, Any] = {
        "status": "rgb_resolution_budget_condition_complete",
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "condition": asdict(condition),
        "target_set_sha256": TARGET_SET_SHA256,
        "painter_config": asdict(config),
        "completed_target_count": len(summaries),
        "targets": summaries,
        "acceptance_checks": {
            "all_five_targets_completed": len(summaries) == len(TARGET_SPECS),
            "all_executed_strokes_improved": all(
                item["every_executed_stroke_improved"] for item in summaries
            ),
            "all_best_frames_not_worse_than_final": all(
                item["best_not_worse_than_final"] for item in summaries
            ),
            "all_frozen_decisions_preserved": all(
                not item["frozen_phase_b0_decision_changed"] for item in summaries
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "training_performed": False,
        "learned_model_used": False,
        "frozen_phase_b0_decision_changed": False,
        "output_directory": str(output_dir),
    }
    aggregate["artifact_sha256"] = _hash_tree(
        output_dir,
        excluded={"aggregate_summary.json", "aggregate_summary.sha256"},
    )
    _write_json(output_dir / "aggregate_summary.json", aggregate)
    (output_dir / "aggregate_summary.sha256").write_text(
        file_sha256(output_dir / "aggregate_summary.json") + "\n",
        encoding="utf-8",
    )
    return aggregate


def _integrity_ok(summary: Mapping[str, Any]) -> bool:
    checks = summary.get("acceptance_checks")
    return bool(
        isinstance(checks, dict)
        and checks.get("all_five_targets_completed") is True
        and checks.get("all_executed_strokes_improved") is True
        and checks.get("all_best_frames_not_worse_than_final") is True
        and checks.get("all_frozen_decisions_preserved") is True
        and summary.get("training_performed") is False
        and summary.get("learned_model_used") is False
        and summary.get("frozen_phase_b0_decision_changed") is False
    )


def evaluate_quantitative_decision(
    baseline_summary: Mapping[str, Any],
    condition_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered quantitative rule without inventing visual judgment."""

    baseline_targets = _target_map(baseline_summary)
    baseline_values = {
        spec.target_id: float(
            baseline_targets[spec.target_id]["high_resolution_final_mse"]
        )
        for spec in TARGET_SPECS
    }
    baseline_mean = fsum(baseline_values.values()) / len(baseline_values)
    records: list[dict[str, Any]] = [
        {
            "condition_id": "A",
            "label": CONDITIONS[0].label,
            "planning_size": 96,
            "maximum_strokes": 210,
            "compute_proxy": CONDITIONS[0].compute_proxy,
            "mean_512_mse": baseline_mean,
            "mean_ratio_to_baseline": 1.0,
            "maximum_target_ratio_to_baseline": 1.0,
            "quantitatively_eligible": False,
            "role": "archived_baseline",
            "targets": [
                {
                    "target_id": spec.target_id,
                    "mse_512": baseline_values[spec.target_id],
                    "ratio_to_baseline": 1.0,
                }
                for spec in TARGET_SPECS
            ],
        }
    ]

    eligible: list[dict[str, Any]] = []
    for condition in NEW_CONDITIONS:
        if condition.condition_id not in condition_summaries:
            raise ValueError(f"Missing condition summary: {condition.condition_id}")
        summary = condition_summaries[condition.condition_id]
        if summary.get("condition", {}).get("condition_id") != condition.condition_id:
            raise ValueError(f"Condition identity mismatch: {condition.condition_id}")
        targets = _target_map(summary)
        target_rows = []
        ratios = []
        values = []
        for spec in TARGET_SPECS:
            mse = float(targets[spec.target_id]["high_resolution_final_mse"])
            ratio = mse / baseline_values[spec.target_id]
            values.append(mse)
            ratios.append(ratio)
            target_rows.append(
                {
                    "target_id": spec.target_id,
                    "mse_512": mse,
                    "baseline_mse_512": baseline_values[spec.target_id],
                    "ratio_to_baseline": ratio,
                    "relative_change": ratio - 1.0,
                }
            )
        mean_mse = fsum(values) / len(values)
        mean_ratio = mean_mse / baseline_mean
        integrity = _integrity_ok(summary)
        qualifies = bool(
            integrity
            and mean_ratio <= MEAN_IMPROVEMENT_RATIO
            and max(ratios) <= MAX_TARGET_WORSENING_RATIO
        )
        record = {
            "condition_id": condition.condition_id,
            "label": condition.label,
            "planning_size": condition.planning_size,
            "maximum_strokes": condition.total_strokes,
            "compute_proxy": condition.compute_proxy,
            "mean_512_mse": mean_mse,
            "mean_ratio_to_baseline": mean_ratio,
            "maximum_target_ratio_to_baseline": max(ratios),
            "integrity_passed": integrity,
            "mean_improvement_threshold_passed": (
                mean_ratio <= MEAN_IMPROVEMENT_RATIO
            ),
            "per_target_worsening_threshold_passed": (
                max(ratios) <= MAX_TARGET_WORSENING_RATIO
            ),
            "quantitatively_eligible": qualifies,
            "role": "new_condition",
            "targets": target_rows,
        }
        records.append(record)
        if qualifies:
            eligible.append(record)

    selected: str | None = None
    provisional_decision = "retain_archived_baseline"
    qualitative_review_required = False
    if eligible:
        best_mean = min(float(record["mean_512_mse"]) for record in eligible)
        near_best = [
            record
            for record in eligible
            if float(record["mean_512_mse"])
            <= best_mean * BEST_MEAN_TOLERANCE_RATIO
        ]
        chosen = min(
            near_best,
            key=lambda record: (
                int(record["compute_proxy"]),
                float(record["mean_512_mse"]),
                str(record["condition_id"]),
            ),
        )
        selected = str(chosen["condition_id"])
        provisional_decision = "meaningful_improvement_pending_qualitative_review"
        qualitative_review_required = True

    means = {str(record["condition_id"]): float(record["mean_512_mse"]) for record in records}
    return {
        "baseline_mean_512_mse": baseline_mean,
        "required_mean_512_mse": baseline_mean * MEAN_IMPROVEMENT_RATIO,
        "conditions": records,
        "factor_effects_on_mean_512_mse": {
            "budget_effect_at_96": means["B"] - means["A"],
            "resolution_effect_at_210": means["C"] - means["A"],
            "combined_effect": means["D"] - means["A"],
            "interaction": means["D"] - means["C"] - means["B"] + means["A"],
        },
        "quantitatively_eligible_conditions": [
            str(record["condition_id"]) for record in eligible
        ],
        "selected_condition_pending_visual_review": selected,
        "provisional_decision": provisional_decision,
        "qualitative_review_required": qualitative_review_required,
        "final_decision": None if qualitative_review_required else provisional_decision,
    }


def marginal_gain_from_progress(path: Path, tail_size: int = 25) -> dict[str, Any]:
    if tail_size < 1:
        raise ValueError("tail_size must be positive.")
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Progress file has no rows: {path}")
    used = min(tail_size, len(rows))
    start = float(rows[-used]["mse_before"])
    end = float(rows[-1]["mse_after"])
    gain = start - end
    return {
        "requested_tail_size": tail_size,
        "used_tail_size": used,
        "start_mse": start,
        "end_mse": end,
        "absolute_gain": gain,
        "relative_gain_from_tail_start": gain / start if start > 0.0 else 0.0,
    }


def _tail_gain_report(
    baseline_dir: Path,
    condition_dirs: Mapping[str, Path],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    roots = {"A": baseline_dir, **condition_dirs}
    for condition_id, root in roots.items():
        report[condition_id] = {
            spec.target_id: marginal_gain_from_progress(
                root / spec.target_id / "progress.csv"
            )
            for spec in TARGET_SPECS
        }
    return report


def _write_comparison_csv(path: Path, decision: Mapping[str, Any]) -> None:
    fields = (
        "condition_id",
        "label",
        "planning_size",
        "maximum_strokes",
        "target_id",
        "mse_512",
        "baseline_mse_512",
        "ratio_to_baseline",
        "relative_change",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition in decision["conditions"]:
            for target in condition["targets"]:
                writer.writerow(
                    {
                        "condition_id": condition["condition_id"],
                        "label": condition["label"],
                        "planning_size": condition["planning_size"],
                        "maximum_strokes": condition["maximum_strokes"],
                        "target_id": target["target_id"],
                        "mse_512": target["mse_512"],
                        "baseline_mse_512": target.get(
                            "baseline_mse_512", target["mse_512"]
                        ),
                        "ratio_to_baseline": target["ratio_to_baseline"],
                        "relative_change": target.get("relative_change", 0.0),
                    }
                )


def _save_mean_plot(path: Path, decision: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = decision["conditions"]
    labels = [
        f"{item['condition_id']}\n{item['planning_size']}px/{item['maximum_strokes']}"
        for item in conditions
    ]
    values = [float(item["mean_512_mse"]) for item in conditions]
    colors = ["#777777", "#4C78A8", "#F58518", "#54A24B"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, values, color=colors)
    axis.axhline(
        float(decision["required_mean_512_mse"]),
        color="#E45756",
        linestyle="--",
        label="10% improvement threshold",
    )
    axis.set_ylabel("Mean common-target 512 RGB MSE")
    axis.set_title("Resolution × stroke-budget ablation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_comparison_montage(
    path: Path,
    *,
    baseline_dir: Path,
    condition_dirs: Mapping[str, Path],
) -> None:
    items: list[tuple[str, Image.Image]] = []
    for spec in TARGET_SPECS:
        with Image.open(baseline_dir / spec.target_id / "processed_target_512.png") as image:
            items.append((f"{spec.target_id} target", image.convert("RGB").copy()))
        with Image.open(baseline_dir / spec.target_id / "final_512.png") as image:
            items.append((f"{spec.target_id} A 96/210", image.convert("RGB").copy()))
        for condition in NEW_CONDITIONS:
            with Image.open(
                condition_dirs[condition.condition_id]
                / spec.target_id
                / "final_512.png"
            ) as image:
                items.append(
                    (
                        f"{spec.target_id} {condition.condition_id} "
                        f"{condition.planning_size}/{condition.total_strokes}",
                        image.convert("RGB").copy(),
                    )
                )
    _make_montage(items, columns=5, cell_size=224).save(path)


def run_resolution_budget_ablation(
    *,
    input_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run B/C/D once while reusing and verifying archived condition A."""

    protocol = validate_protocol()
    target_validation = validate_fixed_targets(input_dir)
    baseline_summary, verified_baseline_artifacts = validate_baseline_result(
        baseline_dir,
        verify_artifacts=True,
    )
    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {output_dir}")
    if incomplete.exists():
        raise FileExistsError(f"Preserve existing incomplete output: {incomplete}")

    incomplete.parent.mkdir(parents=True, exist_ok=True)
    incomplete.mkdir()
    started = time.perf_counter()
    try:
        _write_json(incomplete / "protocol_manifest.json", protocol)
        _write_json(incomplete / "fixed_target_manifest.json", target_validation)
        _write_json(
            incomplete / "baseline_verification.json",
            {
                "status": "verified_immutable_rgb_baseline",
                "baseline_directory": str(baseline_dir),
                "aggregate_summary_sha256": BASELINE_AGGREGATE_SHA256,
                "verified_artifact_count": verified_baseline_artifacts,
                "baseline_mean_512_mse": BASELINE_MEAN_512_MSE,
            },
        )

        summaries: dict[str, dict[str, Any]] = {}
        condition_dirs: dict[str, Path] = {}
        for condition in NEW_CONDITIONS:
            condition_dir = incomplete / condition.slug
            condition_dirs[condition.condition_id] = condition_dir
            summaries[condition.condition_id] = _run_condition(
                condition,
                input_dir=input_dir,
                output_dir=condition_dir,
            )

        decision = evaluate_quantitative_decision(baseline_summary, summaries)
        tail_report = _tail_gain_report(baseline_dir, condition_dirs)
        _write_json(incomplete / "quantitative_decision.json", decision)
        _write_json(incomplete / "final_25_stroke_gains.json", tail_report)
        _write_comparison_csv(incomplete / "condition_target_metrics.csv", decision)
        _save_mean_plot(incomplete / "mean_512_mse_comparison.png", decision)
        _save_comparison_montage(
            incomplete / "five_target_ablation_montage.png",
            baseline_dir=baseline_dir,
            condition_dirs=condition_dirs,
        )

        aggregate: dict[str, Any] = {
            "status": "rgb_resolution_budget_ablation_complete",
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "target_set_sha256": TARGET_SET_SHA256,
            "baseline_aggregate_sha256": BASELINE_AGGREGATE_SHA256,
            "verified_baseline_artifact_count": verified_baseline_artifacts,
            "completed_new_condition_count": len(summaries),
            "new_conditions": summaries,
            "quantitative_decision": decision,
            "final_25_stroke_gains": tail_report,
            "runtime_seconds": time.perf_counter() - started,
            "training_performed": False,
            "learned_model_used": False,
            "frozen_phase_b0_decision_changed": False,
            "output_directory": str(output_dir),
            "final_decision_requires_visual_review": bool(
                decision["qualitative_review_required"]
            ),
        }
        aggregate["artifact_sha256"] = _hash_tree(
            incomplete,
            excluded={
                "aggregate_summary.json",
                "aggregate_summary.sha256",
                "failure.json",
            },
        )
        _write_json(incomplete / "aggregate_summary.json", aggregate)
        (incomplete / "aggregate_summary.sha256").write_text(
            file_sha256(incomplete / "aggregate_summary.json") + "\n",
            encoding="utf-8",
        )
        incomplete.rename(output_dir)
        return aggregate
    except Exception as error:
        _write_json(
            incomplete / "failure.json",
            {
                "status": "rgb_resolution_budget_ablation_incomplete",
                "protocol_id": PROTOCOL_ID,
                "error_type": type(error).__name__,
                "error": str(error),
                "training_performed": False,
            },
        )
        raise


def _smoke_result(planning_size: int, max_steps: int) -> dict[str, Any]:
    config = PainterConfig(
        planning_size=planning_size,
        replay_size=max(32, planning_size),
        supersample=1,
        candidates_per_pool=8,
        error_guided_fraction=0.75,
        patience=3,
        min_improvement=1e-9,
        seed=73,
        gif_stride=1,
        max_attempts_per_candidate=100,
        stages=(StageConfig("smoke", max_steps, 0.20, 0.80, 0.15, 0.35),),
    )
    target = Image.new("RGB", (planning_size, planning_size), color=(64, 96, 128))
    first = plan_rgb_target(target, config, target_stream=0)
    second = plan_rgb_target(target, config, target_stream=0)
    deterministic = first.progress == second.progress and np.array_equal(
        np.asarray(first.final_canvas), np.asarray(second.final_canvas)
    )
    monotonic = bool(first.progress) and all(
        row["mse_after"] < row["mse_before"]
        and row["improvement"] > config.min_improvement
        for row in first.progress
    )
    if not deterministic or not monotonic:
        raise RuntimeError("Ablation synthetic smoke validation failed.")
    return {
        "planning_size": planning_size,
        "maximum_steps": max_steps,
        "executed_strokes": len(first.strokes),
        "deterministic": deterministic,
        "monotonic": monotonic,
    }


def validate_only_ablation_report(
    *,
    input_dir: Path,
    baseline_dir: Path,
) -> dict[str, Any]:
    """Validate protocol, inputs, baseline, and code without creating output."""

    protocol = validate_protocol()
    targets = validate_fixed_targets(input_dir)
    _baseline, artifact_count = validate_baseline_result(
        baseline_dir,
        verify_artifacts=True,
    )
    smoke = (
        _smoke_result(24, 2),
        _smoke_result(32, 3),
    )

    import matplotlib
    import PIL

    return {
        "status": "rgb_resolution_budget_ablation_valid_no_outputs",
        "protocol": protocol,
        "target_validation": targets,
        "baseline_verification": {
            "aggregate_summary_sha256": BASELINE_AGGREGATE_SHA256,
            "verified_artifact_count": artifact_count,
            "mean_512_mse": BASELINE_MEAN_512_MSE,
        },
        "synthetic_smoke": list(smoke),
        "dependencies": {
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "output_side_effects": False,
        "training_performed": False,
        "learned_model_used": False,
        "frozen_phase_b0_decision_changed": False,
    }
