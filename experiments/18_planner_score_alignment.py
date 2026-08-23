#!/usr/bin/env python3
"""Guarded development audit of five scores across frozen latent predictors."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from latent_stroke_dynamics.latent_planner import (
    load_latent_planner_config,
    load_latent_predictor_ensembles,
    load_task_latent_resources,
    encode_task_latents,
)
from latent_stroke_dynamics.planner_score_alignment import (
    DEFAULT_SCORE_ALIGNMENT_CONFIG,
    DEVELOPMENT_CANDIDATE_SEEDS,
    DEVELOPMENT_STATE_SEEDS,
    DEVELOPMENT_TARGET_SEEDS,
    PREDICTOR_FAMILIES,
    SCORE_NAMES,
    aggregate_score_audit,
    candidate_score_variants,
    exact_candidate_metrics,
    exact_candidate_scores,
    load_score_alignment_config,
    pixel_error_patch_weights,
    predict_candidate_latents,
    require_score_audit_authorized,
    require_score_audit_outputs_absent,
    score_audit_output_paths,
    select_score_pair,
    validate_closed_resource_references,
    validate_score_audit_runner_request,
    validate_score_audit_summary,
)
from latent_stroke_dynamics.planning import ProposalConfig, propose_strokes, run_planner
from latent_stroke_dynamics.representation_extension import images_to_grayscale_tensor
from latent_stroke_dynamics.renderer import blank_canvas, random_base_canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_SCORE_ALIGNMENT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--development-score-audit", action="store_true")
    return parser.parse_args()


def image_sha256(image: Image.Image) -> str:
    digest = sha256()
    digest.update(image.mode.encode("utf-8"))
    digest.update(np.asarray(image.size, dtype=np.int64).tobytes())
    digest.update(np.asarray(image).tobytes())
    return digest.hexdigest()


def state_bank(
    target: Image.Image,
    seed: int,
    proposal_config: ProposalConfig,
    trajectory_steps: int,
) -> list[tuple[str, str, int, Image.Image]]:
    exact = run_planner(
        target,
        "exact",
        steps=trajectory_steps,
        seed=seed,
        proposal_config=proposal_config,
        capture_frames=True,
    )
    random = run_planner(
        target,
        "random",
        steps=trajectory_steps,
        seed=seed,
        proposal_config=proposal_config,
        capture_frames=True,
    )
    if len(exact.frames) != trajectory_steps + 1 or len(random.frames) != trajectory_steps + 1:
        raise RuntimeError("State-bank trajectories did not preserve every exact frame.")
    states: list[tuple[str, str, int, Image.Image]] = [
        ("blank_000", "blank", 0, blank_canvas(64))
    ]
    for step in (20, 40, 60, 80):
        states.append((f"exact_pixel_{step:03d}", "exact_pixel", step, exact.frames[step]))
    for step in (20, 40, 60, 80):
        states.append((f"random_{step:03d}", "random", step, random.frames[step]))
    return states


def proposal_from_config(config: dict[str, Any]) -> ProposalConfig:
    development = config["development_score_audit"]
    proposal = config["proposal"]
    return ProposalConfig(
        count=development["candidates_per_state"],
        error_guided_fraction=proposal["error_guided_fraction"],
        min_length=proposal["min_length"],
        max_length=proposal["max_length"],
        width_choices=tuple(proposal["width_choices"]),
        value_choices=tuple(proposal["value_choices"]),
    )


def main() -> None:
    args = parse_args()
    if args.validate_only == args.development_score_audit:
        raise ValueError(
            "Choose exactly one of --validate-only or --development-score-audit."
        )

    config = load_score_alignment_config(args.config)
    resources = config["frozen_resources"]
    closed_config = load_latent_planner_config(resources["latent_planner_config"])
    closed_references = validate_closed_resource_references(config, closed_config)
    paths = score_audit_output_paths(config)
    require_score_audit_outputs_absent(paths)

    if args.validate_only:
        result = validate_score_audit_runner_request(config)
        result.update(closed_references)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    require_score_audit_authorized(config)
    autoencoder, statistics = load_task_latent_resources(closed_config)
    ensembles = load_latent_predictor_ensembles(closed_config)
    all_models_frozen = bool(
        not any(parameter.requires_grad for parameter in autoencoder.parameters())
        and not any(
            parameter.requires_grad
            for family in PREDICTOR_FAMILIES
            for item in ensembles[family]
            for parameter in item.model.parameters()
        )
    )
    if not all_models_frozen:
        raise RuntimeError("A Stage A resource is trainable.")

    development = config["development_score_audit"]
    proposal_config = proposal_from_config(config)
    paths.incomplete.mkdir(parents=True)
    targets_root = paths.incomplete / "targets"
    targets_root.mkdir()

    per_state_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    exact_candidate_sets_unique = True

    for target_index, (target_seed, state_seed, candidate_seed) in enumerate(
        zip(
            DEVELOPMENT_TARGET_SEEDS,
            DEVELOPMENT_STATE_SEEDS,
            DEVELOPMENT_CANDIDATE_SEEDS,
            strict=True,
        ),
        start=1,
    ):
        target_id = f"target_{target_index:02d}"
        print(f"Preparing score-audit target {target_index}/8...")
        target_dir = targets_root / target_id
        target_dir.mkdir()
        target = random_base_canvas(
            size=config["canvas_size"],
            prior_strokes=development["target_strokes"],
            rng=np.random.default_rng(target_seed),
        )
        target.save(target_dir / "target.png")
        target_tokens = encode_task_latents(
            autoencoder,
            statistics,
            (target,),
            batch_size=1,
        )
        target_pixels = images_to_grayscale_tensor((target,)).float()
        states = state_bank(
            target,
            state_seed,
            proposal_config,
            development["state_trajectory_steps"],
        )
        if len(states) != development["states_per_target"]:
            raise RuntimeError("Frozen state-bank size changed.")
        states_dir = target_dir / "states"
        states_dir.mkdir()

        for state_index, (state_id, source, source_step, current) in enumerate(states):
            current.save(states_dir / f"{state_id}.png")
            state_rows.append(
                {
                    "target_id": target_id,
                    "target_seed": target_seed,
                    "state_planner_seed": state_seed,
                    "candidate_seed": candidate_seed,
                    "state_index": state_index,
                    "state_id": state_id,
                    "state_source": source,
                    "state_step": source_step,
                    "state_sha256": image_sha256(current),
                }
            )
            rng = np.random.default_rng(
                np.random.SeedSequence([candidate_seed, state_index, 0])
            )
            candidates = propose_strokes(
                current,
                target,
                rng=rng,
                config=proposal_config,
            )
            exact_canvases, exact_scores = exact_candidate_scores(
                current,
                target,
                candidates,
            )
            signatures = {np.asarray(image).tobytes() for image in exact_canvases}
            exact_candidate_sets_unique = bool(
                exact_candidate_sets_unique and len(signatures) == len(candidates)
            )
            current_tokens = encode_task_latents(
                autoencoder,
                statistics,
                (current,),
                batch_size=1,
            )
            patch_weights = pixel_error_patch_weights(current, target)

            for family in PREDICTOR_FAMILIES:
                started = time.perf_counter()
                predicted = predict_candidate_latents(
                    [item.model for item in ensembles[family]],
                    current_tokens,
                    candidates,
                    batch_size=development["prediction_batch_size"],
                )
                variants = candidate_score_variants(
                    predicted,
                    target_tokens,
                    target_pixels,
                    patch_weights,
                    autoencoder,
                    statistics,
                    batch_size=development["prediction_batch_size"],
                    sobel_edge_weight=development["sobel_edge_weight"],
                )
                elapsed = time.perf_counter() - started

                for score_name in SCORE_NAMES:
                    result = variants[score_name]
                    metrics = exact_candidate_metrics(
                        result.aggregate,
                        exact_scores,
                        tolerance=development["exact_rank_tolerance"],
                    )
                    selected_index = int(metrics["selected_index"])
                    per_state_rows.append(
                        {
                            "target_id": target_id,
                            "target_seed": target_seed,
                            "state_planner_seed": state_seed,
                            "candidate_seed": candidate_seed,
                            "state_index": state_index,
                            "state_id": state_id,
                            "state_source": source,
                            "state_step": source_step,
                            "predictor_family": family,
                            "score_name": score_name,
                            **metrics,
                            "per_model_selected_scores": json.dumps(
                                [
                                    float(value)
                                    for value in result.per_model[:, selected_index]
                                ]
                            ),
                            "elapsed_seconds": elapsed,
                        }
                    )
                    for candidate_index in range(len(candidates)):
                        candidate_rows.append(
                            {
                                "target_id": target_id,
                                "state_id": state_id,
                                "predictor_family": family,
                                "score_name": score_name,
                                "candidate_index": candidate_index,
                                "predicted_score": float(
                                    result.aggregate[candidate_index]
                                ),
                                "exact_pixel_mse": float(
                                    exact_scores[candidate_index]
                                ),
                                "selected": candidate_index == selected_index,
                            }
                        )
        print(f"  target {target_index} complete")

    summary = pd.DataFrame(per_state_rows)
    validate_score_audit_summary(summary, config)
    aggregate = aggregate_score_audit(summary)
    selection = select_score_pair(aggregate)
    expected_candidate_rows = (
        len(DEVELOPMENT_TARGET_SEEDS)
        * development["states_per_target"]
        * len(PREDICTOR_FAMILIES)
        * len(SCORE_NAMES)
        * development["candidates_per_state"]
    )
    implementation_integrity_passed = bool(
        all_models_frozen
        and exact_candidate_sets_unique
        and len(candidate_rows) == expected_candidate_rows
        and len(state_rows)
        == len(DEVELOPMENT_TARGET_SEEDS) * development["states_per_target"]
    )
    if not implementation_integrity_passed:
        raise RuntimeError("Development score-audit implementation integrity failed.")

    summary.to_csv(paths.incomplete / "per_state_summary.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(
        paths.incomplete / "candidate_scores.csv",
        index=False,
    )
    pd.DataFrame(state_rows).to_csv(paths.incomplete / "state_bank.csv", index=False)
    aggregate.to_csv(paths.incomplete / "aggregate_summary.csv", index=False)
    (paths.incomplete / "selection.json").write_text(
        json.dumps(
            {
                "status": "development_score_audit_complete",
                "selection": selection,
                "implementation_integrity_passed": True,
                "models_trained_or_finetuned": False,
                "closed_targets_reused": False,
                "historical_results_unchanged": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (paths.incomplete / "run_config.json").write_text(
        json.dumps(
            {
                "status": "planner_score_development_complete_integrity_passed",
                "single_run": True,
                "target_seeds": list(DEVELOPMENT_TARGET_SEEDS),
                "state_planner_seeds": list(DEVELOPMENT_STATE_SEEDS),
                "candidate_seeds": list(DEVELOPMENT_CANDIDATE_SEEDS),
                "states_per_target": development["states_per_target"],
                "candidates_per_state": development["candidates_per_state"],
                "predictor_families": list(PREDICTOR_FAMILIES),
                "scores": list(SCORE_NAMES),
                "selected_pair": selection,
                "frozen_resource_references": closed_references,
                "all_models_frozen": all_models_frozen,
                "exact_candidate_sets_unique": exact_candidate_sets_unique,
                "implementation_integrity_passed": True,
                "models_trained_or_finetuned": False,
                "closed_targets_reused": False,
                "historical_results_unchanged": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    paths.incomplete.rename(paths.final)

    print("\nDevelopment score-audit aggregate\n")
    print(aggregate.to_string(index=False))
    print("\nSelected frozen predictor/score pair\n")
    print(json.dumps(selection, indent=2))
    print(f"\nSaved development artifacts to: {paths.final.resolve()}")
    print("Do not rerun or tune against these development artifacts.")


if __name__ == "__main__":
    main()
