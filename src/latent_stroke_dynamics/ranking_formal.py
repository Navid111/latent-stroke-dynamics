"""Decision logic for the frozen formal ranking-aware latent comparison."""

from __future__ import annotations

from typing import Any, Mapping


def classify_formal_ranking_result(
    *,
    ranking_retrieval: float,
    mse_retrieval: float,
    improvement_vs_identity: float,
    improvement_vs_mean_delta: float,
    positive_every_primary_crowding: bool,
    all_ranking_seeds_beat_identity: bool,
    oracle_retrieval: float,
    implementation_integrity: bool,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered conjunctive formal ranking-rescue decision."""

    retrieval_gain = ranking_retrieval - mse_retrieval
    checks = {
        "retrieval_at_least_50_percent": (
            ranking_retrieval >= float(thresholds["minimum_formal_retrieval"])
        ),
        "retrieval_gain_at_least_10_points": (
            retrieval_gain
            >= float(thresholds["minimum_absolute_retrieval_gain_over_mse"])
        ),
        "improvement_vs_identity_at_least_30_percent": (
            improvement_vs_identity
            >= float(thresholds["minimum_improvement_vs_identity"])
        ),
        "improvement_vs_mean_delta_at_least_30_percent": (
            improvement_vs_mean_delta
            >= float(thresholds["minimum_improvement_vs_mean_delta"])
        ),
        "positive_every_primary_crowding": positive_every_primary_crowding,
        "all_ranking_seeds_beat_identity": all_ranking_seeds_beat_identity,
        "oracle_retrieval_is_100_percent": (
            oracle_retrieval == float(thresholds["required_oracle_retrieval"])
        ),
        "implementation_integrity_passed": implementation_integrity,
    }
    success = all(checks.values())
    return {
        "classification": (
            "formal_ranking_rescue_success"
            if success
            else "formal_ranking_rescue_not_confirmed"
        ),
        "formal_success": success,
        "checks": checks,
        "ranking_retrieval": ranking_retrieval,
        "mse_only_retrieval": mse_retrieval,
        "absolute_retrieval_gain": retrieval_gain,
        "improvement_vs_identity": improvement_vs_identity,
        "improvement_vs_mean_delta": improvement_vs_mean_delta,
    }
