import pandas as pd
import pytest

from latent_stroke_dynamics.retrieval_diagnostics import (
    merge_test_metadata,
    summarize_retrieval,
    summarize_retrieval_by,
    wilson_interval,
)


def _retrieval_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["linear"] * 4,
            "seed": [11] * 4,
            "sample_id": [0, 1, 2, 3],
            "predicted_label": [
                "true",
                "shift_position",
                "true",
                "change_width",
            ],
            "top1_correct": [True, False, True, False],
            "true_margin": [0.2, -0.1, 0.3, -0.2],
            "score_true": [0.1, 0.2, 0.1, 0.3],
            "score_shift_position": [0.3, 0.1, 0.4, 0.4],
            "score_change_width": [0.4, 0.3, 0.5, 0.1],
            "score_change_intensity": [0.5, 0.4, 0.6, 0.5],
        }
    )


def test_wilson_interval_contains_observed_proportion() -> None:
    lower, upper = wilson_interval(22, 96)
    assert lower < 22 / 96 < upper


def test_retrieval_summary_reports_preferences_and_pairwise_wins() -> None:
    summary = summarize_retrieval(_retrieval_frame()).iloc[0]

    assert summary["top1_correct"] == 2
    assert summary["top1_accuracy"] == pytest.approx(0.5)
    assert summary["predicted_true_rate"] == pytest.approx(0.5)
    assert summary["predicted_shift_position_rate"] == pytest.approx(0.25)
    assert summary["predicted_change_width_rate"] == pytest.approx(0.25)
    assert summary["true_beats_shift_position_rate"] == pytest.approx(0.75)
    assert summary["true_beats_change_width_rate"] == pytest.approx(0.75)
    assert summary["true_beats_change_intensity_rate"] == pytest.approx(1.0)


def test_metadata_merge_filters_test_rows_and_groups_by_crowding() -> None:
    metadata = pd.DataFrame(
        {
            "split": ["train", "test", "test", "test", "test"],
            "sample_id": [0, 0, 1, 2, 3],
            "crowding": [99, 0, 0, 5, 5],
            "stroke_width": [1, 1, 2, 3, 4],
            "stroke_value": [0, 0, 32, 64, 96],
            "stroke_length": [0.3, 0.3, 0.4, 0.5, 0.6],
        }
    )
    merged = merge_test_metadata(_retrieval_frame(), metadata)
    grouped = summarize_retrieval_by(merged, "crowding")

    assert len(merged) == 4
    assert set(merged["crowding"]) == {0, 5}
    assert set(grouped["crowding"]) == {0, 5}
    assert set(grouped["examples"]) == {2}
