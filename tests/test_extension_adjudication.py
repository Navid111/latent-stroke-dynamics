from copy import deepcopy

from latent_stroke_dynamics.extension_adjudication import (
    adjudicate_extension,
    adjudicate_representation,
)


def representation_summary(
    name: str,
    *,
    retrieval: float,
    candidate_zero_difference: float,
    runner_oracle_passed: bool,
) -> dict:
    return {
        "representation": name,
        "classification": "runner_label",
        "representation_eligibility_passed": True,
        "improvement_vs_identity": 0.70,
        "improvement_vs_mean_delta": 0.65,
        "counterfactual_top1_accuracy": retrieval,
        "positive_improvement_every_crowding": True,
        "all_model_seeds_beat_identity": True,
        "all_encoded_counterfactuals_unique": True,
        "all_metrics_finite": True,
        "parameters_within_cap": True,
        "overfit_check": {"loss_decreased": True},
        "exact_target_oracle": {
            "top1_accuracy": 1.0,
            "maximum_candidate_zero_difference": candidate_zero_difference,
            "passed": runner_oracle_passed,
        },
    }


def test_written_oracle_uses_retrieval_and_uniqueness_not_bit_equality() -> None:
    result = adjudicate_representation(
        representation_summary(
            "task_autoencoder",
            retrieval=0.3788888888888889,
            candidate_zero_difference=1.7404556274414062e-05,
            runner_oracle_passed=False,
        )
    )
    assert result["protocol_integrity_passed"] is True
    assert result["adjudicated_classification"] == (
        "average_predictable_but_not_action_usable"
    )
    assert result["candidate_zero_bit_equality_was_frozen_protocol_requirement"] is False


def test_at_or_below_35_percent_retrieval_is_not_predictively_usable() -> None:
    result = adjudicate_representation(
        representation_summary(
            "vit_mae",
            retrieval=0.07111111111111111,
            candidate_zero_difference=0.0,
            runner_oracle_passed=True,
        )
    )
    assert result["protocol_integrity_passed"] is True
    assert result["adjudicated_classification"] == "not_predictively_usable"
    assert "four_way_retrieval_at_or_below_35_percent" in result[
        "classification_reasons"
    ]


def test_top_level_adjudication_does_not_change_source_summary() -> None:
    source = {
        "single_authorized_run": True,
        "do_not_rerun_or_retune": True,
        "implementation_integrity_passed": False,
        "autoencoder": {
            "implementation_integrity_passed": True,
            "protocol_eligibility_passed": True,
        },
        "representations": {
            "task_autoencoder": representation_summary(
                "task_autoencoder",
                retrieval=0.3788888888888889,
                candidate_zero_difference=1.7404556274414062e-05,
                runner_oracle_passed=False,
            ),
            "vit_mae": representation_summary(
                "vit_mae",
                retrieval=0.07111111111111111,
                candidate_zero_difference=0.0,
                runner_oracle_passed=True,
            ),
        },
    }
    before = deepcopy(source)
    result = adjudicate_extension(source)
    assert source == before
    assert result["scientific_metrics_recomputed"] is False
    assert result["written_protocol_global_integrity_passed"] is True
