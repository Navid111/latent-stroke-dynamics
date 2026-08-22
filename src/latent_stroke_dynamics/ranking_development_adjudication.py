"""No-rerun adjudication for the completed ranking-aware development grid."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "full_patch_mse",
    "action_region_mse",
    "outside_region_mse",
    "action_region_next_cosine_distance",
]
RETRIEVAL_COLUMNS = [
    "true_margin",
    "score_true",
    "score_shift_position",
    "score_change_width",
    "score_change_intensity",
]
COMMON_HISTORY_COLUMNS = [
    "seed",
    "epoch",
    "train_balanced_mse",
    "validation_balanced_mse",
]
RANKING_HISTORY_COLUMNS = [
    "ranking_weight",
    "temperature",
    "train_total",
    "train_ranking_cross_entropy",
    "validation_total",
    "validation_ranking_cross_entropy",
]


def finite_columns(frame: pd.DataFrame, columns: list[str]) -> bool:
    """Return whether required columns exist and contain only finite numbers."""

    if frame.empty or not set(columns).issubset(frame.columns):
        return False
    values = frame[columns].to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


def adjudicate_history_finiteness(history: pd.DataFrame) -> dict[str, Any]:
    """Check only columns applicable to each training method.

    The raw runner concatenated MSE-only and ranking-aware histories. Pandas
    represented method-inapplicable ranking columns on MSE-only rows as NaN,
    then a whole-table finite check incorrectly treated those structural blanks
    as non-finite losses.
    """

    if "method" not in history.columns or history.empty:
        return {
            "passed": False,
            "reason": "missing_or_empty_method_column",
        }
    methods = set(history["method"].dropna().astype(str))
    if methods != {"mse_only", "ranking_aware"}:
        return {
            "passed": False,
            "reason": "unexpected_training_methods",
            "methods": sorted(methods),
        }
    mse_rows = history.loc[history["method"] == "mse_only"]
    ranking_rows = history.loc[history["method"] == "ranking_aware"]
    common_finite = finite_columns(history, COMMON_HISTORY_COLUMNS)
    ranking_specific_finite = finite_columns(
        ranking_rows,
        RANKING_HISTORY_COLUMNS,
    )
    ranking_columns_present = set(RANKING_HISTORY_COLUMNS).issubset(history.columns)
    structural_blanks_only = bool(
        ranking_columns_present
        and mse_rows[RANKING_HISTORY_COLUMNS].isna().all().all()
    )
    passed = bool(
        not mse_rows.empty
        and not ranking_rows.empty
        and common_finite
        and ranking_specific_finite
        and structural_blanks_only
    )
    return {
        "passed": passed,
        "mse_only_rows": int(len(mse_rows)),
        "ranking_aware_rows": int(len(ranking_rows)),
        "common_history_columns_finite": common_finite,
        "ranking_specific_columns_finite": ranking_specific_finite,
        "mse_only_ranking_columns_are_expected_structural_blanks": (
            structural_blanks_only
        ),
        "raw_whole_numeric_table_check_was_valid": False,
    }


def adjudicate_development(
    summary: Mapping[str, Any],
    metrics: pd.DataFrame,
    retrieval: pd.DataFrame,
    history: pd.DataFrame,
) -> dict[str, Any]:
    """Apply integrity checks without training or recomputing scientific metrics."""

    if summary.get("development_only") is not True:
        raise ValueError("Source summary is not the completed development grid.")
    if summary.get("do_not_rerun_development") is not True:
        raise ValueError("Source summary lacks its no-rerun marker.")
    if summary.get("formal_data_generated") is not False:
        raise ValueError("Source summary reports formal data generation.")
    if summary.get("diagnostic_test_used_for_selection") is not False:
        raise ValueError("Diagnostic-test selection would invalidate development.")
    if summary.get("selection_used_validation_only") is not True:
        raise ValueError("Ranking setting was not selected on validation only.")

    metrics_finite = finite_columns(metrics, METRIC_COLUMNS)
    retrieval_finite = finite_columns(retrieval, RETRIEVAL_COLUMNS)
    history_result = adjudicate_history_finiteness(history)
    oracles = summary.get("protocol_oracles")
    if not isinstance(oracles, Mapping):
        raise ValueError("Source summary lacks protocol oracles.")
    oracles_passed = bool(
        set(oracles) == {"train", "validation", "diagnostic_test"}
        and all(
            isinstance(value, Mapping) and value.get("passed") is True
            for value in oracles.values()
        )
    )
    overfit = summary.get("ranking_overfit_check")
    if not isinstance(overfit, Mapping):
        raise ValueError("Source summary lacks ranking overfit check.")
    checks = {
        "prediction_metrics_finite": metrics_finite,
        "retrieval_metrics_finite": retrieval_finite,
        "method_applicable_history_values_finite": bool(history_result["passed"]),
        "all_encoded_candidates_unique": summary.get(
            "all_encoded_candidates_unique"
        )
        is True,
        "all_predictor_parameter_counts_valid": summary.get(
            "all_predictor_parameter_counts_valid"
        )
        is True,
        "all_protocol_oracles_passed": oracles_passed,
        "ranking_overfit_loss_decreased": overfit.get("loss_decreased") is True,
        "validation_only_selection": True,
        "formal_data_untouched": True,
    }
    integrity = all(checks.values())
    selected = summary.get("selected_ranking_setting")
    if not isinstance(selected, Mapping):
        raise ValueError("Source summary lacks selected ranking setting.")

    return {
        "status": (
            "ranking_development_protocol_adjudicated_without_rerun"
            if integrity
            else "ranking_development_adjudication_failed"
        ),
        "written_protocol_implementation_integrity_passed": integrity,
        "integrity_checks": checks,
        "history_finiteness_adjudication": history_result,
        "raw_runner_implementation_integrity_passed": summary.get(
            "implementation_integrity_passed"
        ),
        "raw_runner_all_metrics_and_histories_finite": summary.get(
            "all_metrics_and_histories_finite"
        ),
        "raw_summary_preserved_unchanged": True,
        "scientific_metrics_recomputed": False,
        "data_generated": False,
        "models_loaded": False,
        "training_performed": False,
        "selected_ranking_setting": dict(selected),
        "mse_only_validation": summary.get("mse_only_validation"),
        "selected_ranking_validation": summary.get(
            "selected_ranking_validation"
        ),
        "mse_only_diagnostic_test": summary.get(
            "mse_only_diagnostic_test"
        ),
        "selected_ranking_diagnostic_test": summary.get(
            "selected_ranking_diagnostic_test"
        ),
        "diagnostic_absolute_top1_gain": summary.get(
            "diagnostic_absolute_top1_gain"
        ),
        "formal_authorized": False,
        "formal_data_generated": False,
        "do_not_rerun_development": True,
        "historical_decisions_unchanged": True,
        "correction": {
            "issue": "heterogeneous_history_structural_nan_reporting",
            "raw_behavior": (
                "The runner concatenated MSE-only and ranking-aware history rows, "
                "then required every numeric DataFrame cell to be finite."
            ),
            "why_false_positive": (
                "Ranking-only columns are intentionally blank on MSE-only rows; "
                "Pandas represents those inapplicable cells as NaN."
            ),
            "written_integrity_rule": (
                "Every loss value applicable to its training method must be finite."
            ),
        },
    }
