from copy import deepcopy

import numpy as np
import pandas as pd

from latent_stroke_dynamics.ranking_development_adjudication import (
    adjudicate_development,
    adjudicate_history_finiteness,
)


def history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "mse_only",
                "seed": 11,
                "epoch": 1,
                "train_balanced_mse": 0.5,
                "validation_balanced_mse": 0.6,
                "ranking_weight": np.nan,
                "temperature": np.nan,
                "train_total": np.nan,
                "train_ranking_cross_entropy": np.nan,
                "validation_total": np.nan,
                "validation_ranking_cross_entropy": np.nan,
            },
            {
                "method": "ranking_aware",
                "seed": 11,
                "epoch": 1,
                "train_balanced_mse": 0.51,
                "validation_balanced_mse": 0.61,
                "ranking_weight": 1.0,
                "temperature": 0.05,
                "train_total": 1.1,
                "train_ranking_cross_entropy": 0.59,
                "validation_total": 1.2,
                "validation_ranking_cross_entropy": 0.59,
            },
        ]
    )


def metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "full_patch_mse": [0.1],
            "action_region_mse": [0.2],
            "outside_region_mse": [0.05],
            "action_region_next_cosine_distance": [0.03],
        }
    )


def retrieval_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "true_margin": [0.01],
            "score_true": [0.1],
            "score_shift_position": [0.2],
            "score_change_width": [0.3],
            "score_change_intensity": [0.4],
        }
    )


def summary_payload() -> dict:
    return {
        "development_only": True,
        "do_not_rerun_development": True,
        "formal_data_generated": False,
        "diagnostic_test_used_for_selection": False,
        "selection_used_validation_only": True,
        "implementation_integrity_passed": False,
        "all_metrics_and_histories_finite": False,
        "all_encoded_candidates_unique": True,
        "all_predictor_parameter_counts_valid": True,
        "protocol_oracles": {
            "train": {"passed": True},
            "validation": {"passed": True},
            "diagnostic_test": {"passed": True},
        },
        "ranking_overfit_check": {"loss_decreased": True},
        "selected_ranking_setting": {
            "model": "ranking_l1_t0p05",
            "ranking_weight": 1.0,
            "temperature": 0.05,
        },
        "diagnostic_absolute_top1_gain": 0.4895833333333333,
    }


def test_structural_history_nans_are_not_nonfinite_losses() -> None:
    result = adjudicate_history_finiteness(history_frame())
    assert result["passed"] is True
    assert result[
        "mse_only_ranking_columns_are_expected_structural_blanks"
    ] is True


def test_required_ranking_history_nonfinite_value_fails() -> None:
    broken = history_frame()
    broken.loc[broken["method"] == "ranking_aware", "validation_total"] = np.inf
    result = adjudicate_history_finiteness(broken)
    assert result["passed"] is False
    assert result["ranking_specific_columns_finite"] is False


def test_complete_no_rerun_adjudication_passes_without_mutating_summary() -> None:
    summary = summary_payload()
    before = deepcopy(summary)
    result = adjudicate_development(
        summary,
        metrics_frame(),
        retrieval_frame(),
        history_frame(),
    )
    assert summary == before
    assert result["written_protocol_implementation_integrity_passed"] is True
    assert result["scientific_metrics_recomputed"] is False
    assert result["training_performed"] is False
