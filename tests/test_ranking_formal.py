import json
from pathlib import Path

from latent_stroke_dynamics.ranking_formal import classify_formal_ranking_result


ROOT = Path(__file__).resolve().parents[1]
COMMAND = ROOT / "configs" / "ranking-aware-latent-formal-command-2026-08-22.json"
SELECTED = ROOT / "configs" / "ranking-aware-latent-selected-setting-2026-08-22.json"
THRESHOLDS = {
    "minimum_formal_retrieval": 0.5,
    "minimum_absolute_retrieval_gain_over_mse": 0.1,
    "minimum_improvement_vs_identity": 0.3,
    "minimum_improvement_vs_mean_delta": 0.3,
    "required_oracle_retrieval": 1.0,
}


def classify(**overrides):
    values = {
        "ranking_retrieval": 0.7,
        "mse_retrieval": 0.3,
        "improvement_vs_identity": 0.6,
        "improvement_vs_mean_delta": 0.5,
        "positive_every_primary_crowding": True,
        "all_ranking_seeds_beat_identity": True,
        "oracle_retrieval": 1.0,
        "implementation_integrity": True,
        "thresholds": THRESHOLDS,
    }
    values.update(overrides)
    return classify_formal_ranking_result(**values)


def test_formal_ranking_success_is_conjunctive() -> None:
    result = classify()
    assert result["formal_success"] is True
    assert result["classification"] == "formal_ranking_rescue_success"
    assert all(result["checks"].values())


def test_formal_ranking_fails_below_retrieval_threshold() -> None:
    result = classify(ranking_retrieval=0.49)
    assert result["formal_success"] is False
    assert result["checks"]["retrieval_at_least_50_percent"] is False


def test_formal_ranking_fails_without_matched_gain() -> None:
    result = classify(ranking_retrieval=0.55, mse_retrieval=0.50)
    assert result["formal_success"] is False
    assert result["checks"]["retrieval_gain_at_least_10_points"] is False


def test_formal_command_is_authorized_after_validation_and_setting_stays_frozen() -> None:
    command = json.loads(COMMAND.read_text(encoding="utf-8"))
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    assert command["status"] == "authorized_after_local_validation"
    assert command["authorized"] is True
    assert command["formal_seeds"] == list(range(20261104, 20261111))
    assert selected["ranking_weight"] == 1.0
    assert selected["temperature"] == 0.05
    assert selected["formal_authorized"] is False
