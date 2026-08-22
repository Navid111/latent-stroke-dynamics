"""Pure post-hoc adjudication of the completed representation extension.

This module consumes only the saved summary. It does not load data or models,
train anything, or recompute scientific metrics.
"""

from __future__ import annotations

from typing import Any, Mapping


ACTION_RETRIEVAL_MINIMUM = 0.50
NOT_USABLE_RETRIEVAL_MAXIMUM = 0.35
AVERAGE_IMPROVEMENT_MINIMUM = 0.30
LOW_IMROVEMENT_MAXIMUM = 0.10


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Expected numeric field {key!r}.")
    return float(value)


def _boolean(payload: Mapping[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean field {key!r}.")
    return value


def adjudicate_representation(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen written protocol to one saved representation summary."""

    oracle = summary.get("exact_target_oracle")
    overfit = summary.get("overfit_check")
    if not isinstance(oracle, Mapping) or not isinstance(overfit, Mapping):
        raise ValueError("Representation summary lacks oracle or overfit metadata.")

    retrieval = _number(summary, "counterfactual_top1_accuracy")
    improvement_identity = _number(summary, "improvement_vs_identity")
    improvement_mean = _number(summary, "improvement_vs_mean_delta")
    oracle_accuracy = _number(oracle, "top1_accuracy")
    candidate_zero_difference = _number(
        oracle,
        "maximum_candidate_zero_difference",
    )
    encoded_unique = _boolean(summary, "all_encoded_counterfactuals_unique")
    checks = {
        "representation_eligible": _boolean(
            summary,
            "representation_eligibility_passed",
        ),
        "oracle_retrieval_is_100_percent": oracle_accuracy == 1.0,
        "encoded_candidates_are_unique": encoded_unique,
        "all_metrics_are_finite": _boolean(summary, "all_metrics_finite"),
        "tiny_overfit_loss_decreased": _boolean(overfit, "loss_decreased"),
        "parameters_within_cap": _boolean(summary, "parameters_within_cap"),
        "all_model_seeds_beat_identity": _boolean(
            summary,
            "all_model_seeds_beat_identity",
        ),
    }
    protocol_integrity = all(checks.values())
    average_error_passed = bool(
        improvement_identity >= AVERAGE_IMPROVEMENT_MINIMUM
        and improvement_mean >= AVERAGE_IMPROVEMENT_MINIMUM
    )
    positive_primary_crowding = _boolean(
        summary,
        "positive_improvement_every_crowding",
    )

    reasons: list[str] = []
    if not protocol_integrity:
        reasons.append("written_protocol_integrity_failed")
    if retrieval <= NOT_USABLE_RETRIEVAL_MAXIMUM:
        reasons.append("four_way_retrieval_at_or_below_35_percent")
    if (
        improvement_identity < LOW_IMROVEMENT_MAXIMUM
        or improvement_mean < LOW_IMROVEMENT_MAXIMUM
    ):
        reasons.append("improvement_below_10_percent_vs_a_trivial_baseline")

    action_usable = bool(
        protocol_integrity
        and average_error_passed
        and positive_primary_crowding
        and retrieval >= ACTION_RETRIEVAL_MINIMUM
    )
    if action_usable:
        classification = "action_usable"
    elif reasons:
        classification = "not_predictively_usable"
    elif average_error_passed:
        classification = "average_predictable_but_not_action_usable"
        if retrieval < ACTION_RETRIEVAL_MINIMUM:
            reasons.append("four_way_retrieval_below_50_percent")
        if not positive_primary_crowding:
            reasons.append("primary_crowding_condition_failed")
    else:
        classification = "not_predictively_usable"
        reasons.append("average_error_threshold_failed")

    return {
        "representation": summary.get("representation"),
        "runner_reported_classification": summary.get("classification"),
        "adjudicated_classification": classification,
        "classification_reasons": reasons,
        "protocol_integrity_passed": protocol_integrity,
        "protocol_integrity_checks": checks,
        "average_error_conditions_passed": average_error_passed,
        "positive_improvement_every_primary_crowding": positive_primary_crowding,
        "counterfactual_top1_accuracy": retrieval,
        "improvement_vs_identity": improvement_identity,
        "improvement_vs_mean_delta": improvement_mean,
        "oracle_top1_accuracy": oracle_accuracy,
        "runner_reported_oracle_passed": oracle.get("passed"),
        "maximum_candidate_zero_difference": candidate_zero_difference,
        "candidate_zero_bit_equality_was_frozen_protocol_requirement": False,
        "written_oracle_requirement": "exact-target top-1 retrieval equals 100%",
    }


def adjudicate_extension(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Adjudicate a completed saved summary without recomputation."""

    if summary.get("single_authorized_run") is not True:
        raise ValueError("Source is not the single authorized run summary.")
    if summary.get("do_not_rerun_or_retune") is not True:
        raise ValueError("Source summary lacks the completed-run freeze marker.")
    representations = summary.get("representations")
    autoencoder = summary.get("autoencoder")
    if not isinstance(representations, Mapping) or not isinstance(
        autoencoder,
        Mapping,
    ):
        raise ValueError("Source summary is missing representation results.")
    if autoencoder.get("protocol_eligibility_passed") is not True:
        raise ValueError("The selected task autoencoder was not protocol-eligible.")

    adjudicated = {
        name: adjudicate_representation(value)
        for name, value in representations.items()
        if isinstance(value, Mapping)
    }
    if set(adjudicated) != {"task_autoencoder", "vit_mae"}:
        raise ValueError("Expected task_autoencoder and vit_mae summaries.")
    global_integrity = bool(
        autoencoder.get("implementation_integrity_passed") is True
        and all(
            value["protocol_integrity_passed"]
            for value in adjudicated.values()
        )
    )

    return {
        "status": "full_extension_protocol_adjudicated_without_rerun",
        "scientific_metrics_recomputed": False,
        "data_generated": False,
        "models_loaded": False,
        "training_performed": False,
        "raw_summary_preserved_unchanged": True,
        "raw_runner_global_integrity_passed": summary.get(
            "implementation_integrity_passed"
        ),
        "written_protocol_global_integrity_passed": global_integrity,
        "autoencoder_protocol_eligibility_passed": True,
        "representations": adjudicated,
        "corrections": [
            {
                "issue": "task_autoencoder_oracle_bit_equality_guard",
                "raw_behavior": (
                    "The runner required bit-identical separately batched "
                    "candidate-zero encodings in addition to 100% oracle retrieval."
                ),
                "written_protocol": (
                    "Section 7 requires exact-target oracle retrieval of 100%; "
                    "Section 11 separately requires unique encoded candidates."
                ),
                "adjudication": (
                    "The task oracle retrieved the true candidate for all examples "
                    "and all candidates were unique, so the written oracle and "
                    "integrity requirements passed. The 1.7404556274414062e-05 "
                    "re-encoding difference remains reported as numerical drift."
                ),
            },
            {
                "issue": "classification_precedence_at_35_percent_retrieval",
                "raw_behavior": (
                    "The runner assigned the average-predictable label whenever "
                    "average error passed, even below 35% retrieval."
                ),
                "written_protocol": (
                    "Section 9 says retrieval at or below 35% is not "
                    "predictively usable."
                ),
                "adjudication": (
                    "The written at-or-below-35% rule takes precedence."
                ),
            },
        ],
        "historical_decisions_unchanged": True,
        "do_not_rerun_or_retune": True,
    }
