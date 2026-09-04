"""Fail-closed infrastructure for the fixed straight-versus-curve comparison.

This module validates the frozen procedural target manifest, prepares complete
per-run and aggregate evidence, and refuses comparative execution unless a
separate one-time authorization file matches the exact runner commit.  The
primary comparison trains or loads no learned model.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from hashlib import sha256
import json
from math import fsum
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .quadratic_bezier_extension import (
    PRIMITIVES,
    PROTOCOL_ID,
    SEEDS,
    TARGET_IDS,
    GeneratedTarget,
    PrimitiveName,
    extension_painter_config,
    generate_rights_safe_targets,
    generated_target_manifest,
    image_pixel_sha256,
    plan_primitive_target,
    stroke_to_record,
)
from .quadratic_bezier_replay import replay_extension_strokes_high_resolution
from .rgb_coarse_to_fine import (
    PainterConfig,
    StageConfig,
    _hash_tree,
    _make_montage,
    _write_json,
    file_sha256,
    pixel_mae,
    pixel_mse,
)


FREEZE_MANIFEST_PATH = Path(
    "configs/quadratic-bezier-target-freeze-2026-09-04.json"
)
FROZEN_TARGET_SET_SHA256 = (
    "26bada941bfd8f49f09333d70d397364e82f5ddbb6e1228324f24fb9d2b30bfd"
)
FROZEN_TARGET_HASHES = {
    "01_ring_symbol": "548ee3d03644308c066f11be64234db8a28a67e0a24cb5d21bec6bd6aab4940b",
    "02_curved_glyph": "b2e0036f2eb1b4275fb553ae970ab2a93ffb08b229b21b7ebe2dd194e2d0a7da",
    "03_organic_silhouette": "838edf87ab05d3b289af18673c87d51e5cd56f77ebef4aa6bb99c7f7685af398",
    "04_mixed_geometry": "d660b8247c098d4fc3dac9b51330678371d3efaa73406f86cbfacc6762357f66",
    "05_layered_landscape": "4bd0794c85c2be198ee93a2ab0d155ea0758ae0f84bd56fac8428da0e604b45a",
    "06_dense_scene": "79d51333fc6d94b2e269cc3a895c0dba2fe12ca2f01d8e9565ef691614c3e3fe",
}
VALIDATED_IMPLEMENTATION_COMMIT = "7bdcd2e847ca7c5a1faf8a086b26441d8de1a4e1"
VALIDATION_HANDOFF_COMMIT = "f5037d4c64931fa57e403081c7c384845742c7fd"
AUTHORIZATION_STATUS = "authorized_for_one_quadratic_bezier_comparison"
EXPECTED_PAIR_COUNT = len(TARGET_IDS) * len(SEEDS)
EXPECTED_RUN_COUNT = EXPECTED_PAIR_COUNT * len(PRIMITIVES)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _exact_freeze_values() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "freeze_id": "quadratic_bezier_target_freeze_v1",
        "status": "target_hashes_frozen_execution_unauthorized",
        "branch": "quadratic-bezier-extension",
        "base_commit": "d5f1190ab9b62d5adff7fb56c5cc1ffd4d850177",
        "validated_implementation_commit": VALIDATED_IMPLEMENTATION_COMMIT,
        "validation_handoff_commit": VALIDATION_HANDOFF_COMMIT,
        "conditions": list(PRIMITIVES),
        "planning_size": 128,
        "replay_size": 512,
        "accepted_strokes": 420,
        "stage_budgets": [80, 140, 200],
        "candidates_per_pool": 64,
        "error_guided_fraction": 0.8,
        "patience": 12,
        "min_improvement": 1e-9,
        "target_generator": "deterministic_procedural_pillow_v1",
        "target_source": "deterministic_procedural_rights_safe",
        "target_count": 6,
        "target_order": list(TARGET_IDS),
        "target_stream_by_id": {
            target_id: index for index, target_id in enumerate(TARGET_IDS)
        },
        "target_hashes_frozen": True,
        "target_set_sha256": FROZEN_TARGET_SET_SHA256,
        "target_sha256": FROZEN_TARGET_HASHES,
        "seed_order": list(SEEDS),
        "decision_rule": {
            "minimum_mean_mse_improvement_fraction": 0.05,
            "minimum_improved_target_count": 4,
            "maximum_per_target_worsening_ratio": 1.05,
            "blinded_review_required": True,
        },
        "execution_authorized": False,
        "maximum_completed_executions": 1,
        "completed_executions": 0,
        "learned_model_allowed": False,
    }


def validate_target_freeze(
    path: Path = FREEZE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Verify the immutable target, seed, budget, and decision manifest."""

    manifest = _read_json(path)
    for key, expected in _exact_freeze_values().items():
        if manifest.get(key) != expected:
            raise ValueError(f"Frozen comparison manifest mismatch for {key}.")

    evidence = manifest.get("validation_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("Frozen validation evidence is missing.")
    evidence_expected = {
        "pytest_passed": 199,
        "pytest_duration_seconds": 61.78,
        "validation_status": "quadratic_bezier_extension_valid_no_outputs",
        "validated_config_sha256": "fb9cf09e46ae1beb51e2462a93832c80e45e17db811586d7d4e30573c3b5cbfa",
        "dependencies": {"numpy": "2.1.3", "pillow": "11.3.0"},
        "output_side_effects": False,
        "comparative_outputs_viewed": False,
        "training_performed": False,
        "learned_model_used": False,
        "closed_experiments_changed": False,
        "pytest_log": "docs/quadratic-bezier-pytest-2026-09-04.txt",
        "validation_report": "docs/quadratic-bezier-validation-2026-09-04.json",
    }
    if evidence != evidence_expected:
        raise ValueError("Frozen validation evidence changed.")

    generated = generated_target_manifest(512)
    if generated.get("target_set_sha256") != FROZEN_TARGET_SET_SHA256:
        raise ValueError("Generated ordered target-set hash does not match the freeze.")
    entries = generated.get("targets")
    if not isinstance(entries, list):
        raise ValueError("Generated target manifest has no target list.")
    observed_ids = [item.get("target_id") for item in entries]
    observed_hashes = {
        str(item.get("target_id")): item.get("pixel_sha256") for item in entries
    }
    if observed_ids != list(TARGET_IDS):
        raise ValueError("Generated target order does not match the freeze.")
    if observed_hashes != FROZEN_TARGET_HASHES:
        raise ValueError("Generated target pixels do not match the frozen hashes.")

    for seed in SEEDS:
        painter = extension_painter_config(seed)
        if painter.planning_size != 128 or painter.replay_size != 512:
            raise ValueError("Frozen comparison resolution changed.")
        if painter.candidates_per_pool != 64:
            raise ValueError("Frozen candidate-pool size changed.")
        if [stage.max_steps for stage in painter.stages] != [80, 140, 200]:
            raise ValueError("Frozen stage budgets changed.")
    return manifest


def validate_execution_authorization(
    path: Path,
    *,
    source_commit: str,
    freeze_path: Path = FREEZE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Validate a separate one-time authorization before any output is created."""

    authorization = _read_json(path)
    if authorization.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("Authorization protocol mismatch.")
    if authorization.get("status") != AUTHORIZATION_STATUS:
        raise PermissionError("Comparison execution is not authorized.")
    if authorization.get("execution_authorized") is not True:
        raise PermissionError("Comparison execution is not authorized.")
    if authorization.get("maximum_completed_executions") != 1:
        raise PermissionError("Authorization must allow exactly one completion.")
    if authorization.get("completed_executions") != 0:
        raise PermissionError("The one-time authorization has already been consumed.")
    if authorization.get("learned_model_allowed") is not False:
        raise PermissionError("A learned model is prohibited in this comparison.")
    if authorization.get("target_set_sha256") != FROZEN_TARGET_SET_SHA256:
        raise PermissionError("Authorization target-set hash mismatch.")
    if authorization.get("target_freeze_sha256") != file_sha256(freeze_path):
        raise PermissionError("Authorization target-freeze file mismatch.")
    if authorization.get("authorized_runner_commit") != source_commit:
        raise PermissionError("Authorization does not match the checked-out runner commit.")
    output_name = authorization.get("expected_output_name")
    if not isinstance(output_name, str) or not output_name.strip():
        raise PermissionError("Authorization has no fixed output-directory name.")
    return authorization


def _smoke_config(seed: int) -> PainterConfig:
    return PainterConfig(
        planning_size=32,
        replay_size=32,
        supersample=1,
        candidates_per_pool=8,
        error_guided_fraction=0.75,
        patience=3,
        min_improvement=1e-9,
        seed=seed,
        gif_stride=1,
        max_attempts_per_candidate=100,
        stages=(StageConfig("smoke", 3, 0.20, 0.80, 0.12, 0.30),),
    )


def validate_only_comparison_report(
    freeze_path: Path = FREEZE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Validate all fixed inputs and tiny planners without creating outputs."""

    freeze = validate_target_freeze(freeze_path)
    source = generate_rights_safe_targets(128)[3].image.resize(
        (32, 32), Image.Resampling.LANCZOS
    )
    smoke: dict[str, Any] = {}
    for primitive in PRIMITIVES:
        config = _smoke_config(73)
        first = plan_primitive_target(source, primitive, config, target_stream=3)
        second = plan_primitive_target(source, primitive, config, target_stream=3)
        deterministic = (
            first.progress == second.progress
            and first.strokes == second.strokes
            and np.array_equal(
                np.asarray(first.final_canvas), np.asarray(second.final_canvas)
            )
        )
        monotonic = bool(first.progress) and all(
            float(row["mse_after"]) < float(row["mse_before"])
            and float(row["improvement"]) > config.min_improvement
            for row in first.progress
        )
        if not deterministic or not monotonic:
            raise RuntimeError(f"{primitive} runner smoke validation failed.")
        smoke[primitive] = {
            "executed_strokes": len(first.strokes),
            "candidate_pools": sum(
                int(item["candidate_pools"]) for item in first.stage_stats
            ),
            "initial_mse": first.initial_mse,
            "final_mse": first.final_mse,
            "deterministic": deterministic,
            "monotonic": monotonic,
        }

    import matplotlib
    import PIL

    return {
        "status": "quadratic_bezier_comparison_runner_valid_no_outputs",
        "protocol_id": PROTOCOL_ID,
        "target_freeze_sha256": file_sha256(freeze_path),
        "target_set_sha256": freeze["target_set_sha256"],
        "target_count": len(TARGET_IDS),
        "seed_count": len(SEEDS),
        "condition_count": len(PRIMITIVES),
        "expected_pair_count": EXPECTED_PAIR_COUNT,
        "expected_run_count": EXPECTED_RUN_COUNT,
        "synthetic_smoke": smoke,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "output_side_effects": False,
        "execution_authorized": False,
        "comparative_outputs_viewed": False,
        "training_performed": False,
        "learned_model_used": False,
        "closed_experiments_changed": False,
    }


def _stage_end_steps(stage_stats: Sequence[Mapping[str, Any]]) -> set[int]:
    steps: set[int] = set()
    cumulative = 0
    for item in stage_stats:
        cumulative += int(item["executed_steps"])
        if cumulative > 0:
            steps.add(cumulative)
    return steps


def _write_progress_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "step",
        "stage",
        "stage_step",
        "candidate_pool",
        "selected_candidate_index",
        "primitive",
        "action_json",
        "mse_before",
        "mse_after",
        "best_mse",
        "improvement",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "step": row["step"],
                    "stage": row["stage"],
                    "stage_step": row["stage_step"],
                    "candidate_pool": row["candidate_pool"],
                    "selected_candidate_index": row["selected_candidate_index"],
                    "primitive": row["primitive"],
                    "action_json": json.dumps(
                        row["action"], sort_keys=True, separators=(",", ":")
                    ),
                    "mse_before": row["mse_before"],
                    "mse_after": row["mse_after"],
                    "best_mse": row["best_mse"],
                    "improvement": row["improvement"],
                }
            )


def run_target_seed_condition(
    target: GeneratedTarget,
    *,
    primitive: PrimitiveName,
    seed: int,
    target_stream: int,
    output_dir: Path,
    config: PainterConfig | None = None,
) -> dict[str, Any]:
    """Execute and archive one target-seed-primitive run."""

    config = config or extension_painter_config(seed)
    config.validate()
    if primitive not in PRIMITIVES:
        raise ValueError(f"Unsupported primitive: {primitive}")
    if seed != config.seed:
        raise ValueError("Run seed and painter seed differ.")
    if target.target_id != TARGET_IDS[target_stream]:
        raise ValueError("Target and frozen target-stream mapping differ.")
    if target.image.mode != "RGB" or target.image.size != (
        config.replay_size,
        config.replay_size,
    ):
        raise ValueError("Run target must be the frozen replay-size RGB image.")

    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    target_512 = target.image.copy()
    target_128 = target_512.resize(
        (config.planning_size, config.planning_size),
        Image.Resampling.LANCZOS,
    )
    target_128.save(output_dir / "target_128.png")
    target_512.save(output_dir / "target_512.png")

    result = plan_primitive_target(
        target_128,
        primitive,
        config,
        target_stream=target_stream,
    )
    result.initial_canvas.save(output_dir / "initial_128.png")
    result.best_canvas.save(output_dir / "best_128.png")
    result.final_canvas.save(output_dir / "final_128.png")

    best_512, final_512, _frames = replay_extension_strokes_high_resolution(
        result.strokes,
        planning_size=config.planning_size,
        output_size=config.replay_size,
        supersample=config.supersample,
        best_step=result.best_step,
        capture_steps=_stage_end_steps(result.stage_stats),
        gif_stride=config.gif_stride,
    )
    best_512.save(output_dir / "best_512.png")
    final_512.save(output_dir / "final_512.png")

    actions = [stroke_to_record(stroke) for stroke in result.strokes]
    _write_json(output_dir / "actions.json", actions)
    _write_progress_csv(output_dir / "progress.csv", result.progress)
    _write_json(
        output_dir / "run_config.json",
        {
            "protocol_id": PROTOCOL_ID,
            "target_id": target.target_id,
            "target_stream": target_stream,
            "target_pixel_sha256": image_pixel_sha256(target_512),
            "primitive": primitive,
            "seed": seed,
            "painter_config": asdict(config),
            "selection": "exact_rendered_rgb_target_pixel_mse",
            "training_performed": False,
            "learned_model_used": False,
        },
    )

    monotonic = bool(result.progress) and all(
        float(row["mse_after"]) < float(row["mse_before"])
        and float(row["improvement"]) > config.min_improvement
        for row in result.progress
    )
    candidate_pools = sum(
        int(item["candidate_pools"]) for item in result.stage_stats
    )
    summary: dict[str, Any] = {
        "status": "quadratic_bezier_comparison_run_complete",
        "protocol_id": PROTOCOL_ID,
        "target_id": target.target_id,
        "category": target.category,
        "provenance": target.provenance,
        "target_stream": target_stream,
        "target_pixel_sha256": image_pixel_sha256(target_512),
        "primitive": primitive,
        "seed": seed,
        "planning_size": config.planning_size,
        "replay_size": config.replay_size,
        "initial_mse": result.initial_mse,
        "best_mse": result.best_mse,
        "final_mse": result.final_mse,
        "final_mae": result.final_mae,
        "best_step": result.best_step,
        "executed_strokes": len(result.strokes),
        "maximum_strokes": sum(stage.max_steps for stage in config.stages),
        "candidate_pools": candidate_pools,
        "candidates_per_pool": config.candidates_per_pool,
        "candidate_renders": candidate_pools * config.candidates_per_pool,
        "stage_stats": list(result.stage_stats),
        "every_executed_stroke_improved": monotonic,
        "best_not_worse_than_final": result.best_mse <= result.final_mse,
        "high_resolution_best_mse": pixel_mse(best_512, target_512),
        "high_resolution_final_mse": pixel_mse(final_512, target_512),
        "high_resolution_final_mae": pixel_mae(final_512, target_512),
        "runtime_seconds": time.perf_counter() - started,
        "training_performed": False,
        "learned_model_used": False,
        "closed_experiments_changed": False,
    }
    summary["artifact_sha256"] = _hash_tree(
        output_dir,
        excluded={"summary.json", "summary.sha256"},
    )
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.sha256").write_text(
        file_sha256(output_dir / "summary.json") + "\n",
        encoding="utf-8",
    )
    return summary


def _ratio(numerator: float, denominator: float) -> float:
    if denominator > 0.0:
        return numerator / denominator
    return 1.0 if numerator == 0.0 else float("inf")


def evaluate_quantitative_decision(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    integrity_passed: bool,
) -> dict[str, Any]:
    """Apply the frozen quantitative rule without inventing visual judgment."""

    if len(pair_rows) != EXPECTED_PAIR_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_PAIR_COUNT} target-seed pairs; got {len(pair_rows)}."
        )
    expected_keys = [
        (target_id, seed) for target_id in TARGET_IDS for seed in SEEDS
    ]
    observed_keys = [
        (str(row["target_id"]), int(row["seed"])) for row in pair_rows
    ]
    if observed_keys != expected_keys:
        raise ValueError("Target-seed pair order changed.")

    per_target: list[dict[str, Any]] = []
    for target_id in TARGET_IDS:
        rows = [row for row in pair_rows if row["target_id"] == target_id]
        straight_values = [float(row["straight_mse_512"]) for row in rows]
        curve_values = [float(row["quadratic_bezier_mse_512"]) for row in rows]
        straight_mean = fsum(straight_values) / len(straight_values)
        curve_mean = fsum(curve_values) / len(curve_values)
        ratio = _ratio(curve_mean, straight_mean)
        per_target.append(
            {
                "target_id": target_id,
                "seed_count": len(rows),
                "straight_mean_mse_512": straight_mean,
                "quadratic_bezier_mean_mse_512": curve_mean,
                "curve_to_straight_ratio": ratio,
                "curve_relative_change": ratio - 1.0,
                "curve_improved": curve_mean < straight_mean,
            }
        )

    straight_all = [float(row["straight_mse_512"]) for row in pair_rows]
    curve_all = [float(row["quadratic_bezier_mse_512"]) for row in pair_rows]
    straight_mean = fsum(straight_all) / len(straight_all)
    curve_mean = fsum(curve_all) / len(curve_all)
    overall_ratio = _ratio(curve_mean, straight_mean)
    improvement_fraction = 1.0 - overall_ratio
    improved_target_count = sum(bool(item["curve_improved"]) for item in per_target)
    maximum_target_ratio = max(float(item["curve_to_straight_ratio"]) for item in per_target)

    mean_threshold = improvement_fraction >= 0.05
    target_count_threshold = improved_target_count >= 4
    worsening_threshold = maximum_target_ratio <= 1.05
    materially_eligible = bool(
        integrity_passed
        and curve_mean < straight_mean
        and mean_threshold
        and target_count_threshold
        and worsening_threshold
    )

    if not integrity_passed or curve_mean >= straight_mean:
        provisional = "no_material_improvement"
        qualitative_review_required = False
        final_decision: str | None = provisional
    elif materially_eligible:
        provisional = "material_improvement_pending_blinded_review"
        qualitative_review_required = True
        final_decision = None
    else:
        provisional = "minor_improvement"
        qualitative_review_required = False
        final_decision = provisional

    return {
        "primary_metric": "mean_final_512_rgb_mse_across_target_seed_pairs",
        "pair_count": len(pair_rows),
        "straight_mean_mse_512": straight_mean,
        "quadratic_bezier_mean_mse_512": curve_mean,
        "curve_to_straight_ratio": overall_ratio,
        "curve_improvement_fraction": improvement_fraction,
        "improved_target_count": improved_target_count,
        "maximum_per_target_worsening_ratio": maximum_target_ratio,
        "per_target": per_target,
        "thresholds": {
            "minimum_mean_mse_improvement_fraction": 0.05,
            "minimum_improved_target_count": 4,
            "maximum_per_target_worsening_ratio": 1.05,
            "blinded_review_required": True,
        },
        "checks": {
            "integrity_passed": integrity_passed,
            "mean_improvement_threshold_passed": mean_threshold,
            "improved_target_count_threshold_passed": target_count_threshold,
            "per_target_worsening_threshold_passed": worsening_threshold,
        },
        "quantitatively_materially_eligible": materially_eligible,
        "provisional_decision": provisional,
        "qualitative_review_required": qualitative_review_required,
        "final_decision": final_decision,
    }


def _write_pair_metrics(path: Path, pair_rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "target_id",
        "seed",
        "straight_mse_512",
        "quadratic_bezier_mse_512",
        "curve_to_straight_ratio",
        "curve_relative_change",
        "straight_candidate_renders",
        "quadratic_bezier_candidate_renders",
        "straight_runtime_seconds",
        "quadratic_bezier_runtime_seconds",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in pair_rows:
            writer.writerow({field: row[field] for field in fields})


def _save_mean_plot(path: Path, decision: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [
        float(decision["straight_mean_mse_512"]),
        float(decision["quadratic_bezier_mean_mse_512"]),
    ]
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.bar(["Straight line", "Quadratic Bezier"], values, color=["#4C78A8", "#F58518"])
    axis.set_ylabel("Mean final 512 RGB MSE")
    axis.set_title("Matched primitive comparison")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_target_ratio_plot(path: Path, decision: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = decision["per_target"]
    labels = [str(item["target_id"]).split("_", 1)[0] for item in rows]
    ratios = [float(item["curve_to_straight_ratio"]) for item in rows]
    colors = ["#54A24B" if value < 1.0 else "#E45756" for value in ratios]
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(labels, ratios, color=colors)
    axis.axhline(1.0, color="#333333", linestyle="-", linewidth=1.0)
    axis.axhline(1.05, color="#E45756", linestyle="--", linewidth=1.0, label="5% worsening guard")
    axis.set_xlabel("Frozen target index")
    axis.set_ylabel("Curve / straight mean 512 MSE")
    axis.set_title("Per-target primitive ratio across three seeds")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _load_progress_values(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Progress file has no rows: {path}")
    return [float(rows[0]["mse_before"])] + [
        float(row["mse_after"]) for row in rows
    ]


def _save_aggregate_progress_plot(path: Path, runs_root: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    for primitive, color in zip(PRIMITIVES, ("#4C78A8", "#F58518"), strict=True):
        trajectories = []
        for target_id in TARGET_IDS:
            for seed in SEEDS:
                values = _load_progress_values(
                    runs_root / target_id / f"seed_{seed}" / primitive / "progress.csv"
                )
                trajectories.append(values)
        maximum = max(len(values) for values in trajectories)
        padded = [
            values + [values[-1]] * (maximum - len(values)) for values in trajectories
        ]
        mean_values = np.asarray(padded, dtype=np.float64).mean(axis=0)
        axis.plot(
            list(range(len(mean_values))),
            mean_values,
            color=color,
            label=primitive.replace("_", " "),
        )
    axis.set_xlabel("Accepted stroke")
    axis.set_ylabel("Mean planning-resolution RGB MSE")
    axis.set_title("Aggregate exact-pixel planning trajectories")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _blinded_primitives(target_id: str, seed: int) -> tuple[str, str]:
    payload = f"{PROTOCOL_ID}|{target_id}|{seed}|blinded-v1".encode("utf-8")
    if sha256(payload).digest()[0] % 2 == 0:
        return ("straight", "quadratic_bezier")
    return ("quadratic_bezier", "straight")


def _save_blinded_review_materials(
    root: Path,
    *,
    targets: Sequence[GeneratedTarget],
) -> dict[str, Any]:
    items: list[tuple[str, Image.Image]] = []
    mapping: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    pair_index = 0
    for target in targets:
        for seed in SEEDS:
            pair_index += 1
            pair_id = f"P{pair_index:02d}"
            primitive_a, primitive_b = _blinded_primitives(target.target_id, seed)
            items.append((f"{pair_id} target", target.image.copy()))
            for label, primitive in (("A", primitive_a), ("B", primitive_b)):
                path = (
                    root
                    / "runs"
                    / target.target_id
                    / f"seed_{seed}"
                    / primitive
                    / "final_512.png"
                )
                with Image.open(path) as image:
                    items.append((f"{pair_id} method {label}", image.convert("RGB").copy()))
            mapping.append(
                {
                    "pair_id": pair_id,
                    "target_id": target.target_id,
                    "seed": seed,
                    "method_A": primitive_a,
                    "method_B": primitive_b,
                }
            )
            review_rows.append(
                {
                    "pair_id": pair_id,
                    "preferred_method": "",
                    "systematic_regression": "",
                    "notes": "",
                }
            )

    _make_montage(items, columns=3, cell_size=192).save(
        root / "blinded_review_montage.png"
    )
    _write_json(root / "blinded_mapping_do_not_open_before_review.json", mapping)
    with (root / "blinded_review_sheet.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "pair_id",
                "preferred_method",
                "systematic_regression",
                "notes",
            ),
        )
        writer.writeheader()
        writer.writerows(review_rows)
    return {
        "pair_count": len(mapping),
        "montage": "blinded_review_montage.png",
        "review_sheet": "blinded_review_sheet.csv",
        "mapping": "blinded_mapping_do_not_open_before_review.json",
        "mapping_rule": "Deterministic SHA-256 assignment independent of outcomes.",
    }


def run_fixed_comparison(
    *,
    output_dir: Path,
    authorization_path: Path,
    source_commit: str,
    freeze_path: Path = FREEZE_MANIFEST_PATH,
) -> dict[str, Any]:
    """Run the frozen 36-run comparison exactly once when separately authorized."""

    freeze = validate_target_freeze(freeze_path)
    authorization = validate_execution_authorization(
        authorization_path,
        source_commit=source_commit,
        freeze_path=freeze_path,
    )
    if output_dir.name != authorization["expected_output_name"]:
        raise PermissionError("Output directory does not match the authorization.")

    incomplete = output_dir.with_name(output_dir.name + ".incomplete")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {output_dir}")
    if incomplete.exists():
        raise FileExistsError(f"Preserve existing incomplete output: {incomplete}")

    incomplete.parent.mkdir(parents=True, exist_ok=True)
    incomplete.mkdir()
    started = time.perf_counter()
    try:
        _write_json(incomplete / "target_freeze_manifest.json", freeze)
        _write_json(incomplete / "execution_authorization.json", authorization)
        _write_json(
            incomplete / "environment_manifest.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "source_commit": source_commit,
                "target_freeze_sha256": file_sha256(freeze_path),
                "authorization_sha256": file_sha256(authorization_path),
            },
        )

        targets = generate_rights_safe_targets(512)
        target_root = incomplete / "targets"
        target_root.mkdir()
        for target in targets:
            if image_pixel_sha256(target.image) != FROZEN_TARGET_HASHES[target.target_id]:
                raise ValueError(f"Frozen target changed: {target.target_id}")
            target.image.save(target_root / f"{target.target_id}.png")

        runs_root = incomplete / "runs"
        summaries: list[dict[str, Any]] = []
        pair_rows: list[dict[str, Any]] = []
        for target_stream, target in enumerate(targets):
            for seed in SEEDS:
                pair: dict[str, dict[str, Any]] = {}
                for primitive in PRIMITIVES:
                    run_dir = (
                        runs_root
                        / target.target_id
                        / f"seed_{seed}"
                        / primitive
                    )
                    summary = run_target_seed_condition(
                        target,
                        primitive=primitive,
                        seed=seed,
                        target_stream=target_stream,
                        output_dir=run_dir,
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
                        "target_id": target.target_id,
                        "seed": seed,
                        "straight_mse_512": straight_mse,
                        "quadratic_bezier_mse_512": curve_mse,
                        "curve_to_straight_ratio": ratio,
                        "curve_relative_change": ratio - 1.0,
                        "straight_candidate_renders": straight["candidate_renders"],
                        "quadratic_bezier_candidate_renders": curve["candidate_renders"],
                        "straight_runtime_seconds": straight["runtime_seconds"],
                        "quadratic_bezier_runtime_seconds": curve["runtime_seconds"],
                    }
                )

        integrity = {
            "all_expected_runs_completed": len(summaries) == EXPECTED_RUN_COUNT,
            "all_expected_pairs_completed": len(pair_rows) == EXPECTED_PAIR_COUNT,
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
            incomplete / "aggregate_progress_by_primitive.png", runs_root
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
        aggregate: dict[str, Any] = {
            "status": "quadratic_bezier_fixed_comparison_complete",
            "protocol_id": PROTOCOL_ID,
            "source_commit": source_commit,
            "authorization_id": authorization.get("authorization_id"),
            "target_set_sha256": FROZEN_TARGET_SET_SHA256,
            "target_freeze_sha256": file_sha256(freeze_path),
            "completed_run_count": len(summaries),
            "completed_pair_count": len(pair_rows),
            "run_summaries": summaries,
            "pair_metrics": pair_rows,
            "integrity": integrity,
            "integrity_passed": integrity_passed,
            "quantitative_decision": decision,
            "blinded_review": blinded,
            "total_candidate_renders": total_candidate_renders,
            "condition_runtime_seconds": total_runtime_seconds,
            "wall_clock_seconds": time.perf_counter() - started,
            "training_performed": False,
            "learned_model_used": False,
            "closed_experiments_changed": False,
            "final_decision_requires_blinded_review": bool(
                decision["qualitative_review_required"]
            ),
            "output_directory": str(output_dir),
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
                "status": "quadratic_bezier_fixed_comparison_incomplete",
                "protocol_id": PROTOCOL_ID,
                "source_commit": source_commit,
                "error_type": type(error).__name__,
                "error": str(error),
                "training_performed": False,
                "learned_model_used": False,
                "closed_experiments_changed": False,
            },
        )
        raise
