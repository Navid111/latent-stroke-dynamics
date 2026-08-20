"""Post-hoc diagnostics for Gate 2 counterfactual retrieval outputs."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

import pandas as pd
from pandas.api.types import is_bool_dtype


CANDIDATE_NAMES: tuple[str, ...] = (
    "true",
    "shift_position",
    "change_width",
    "change_intensity",
)


def _boolean_values(series: pd.Series) -> pd.Series:
    if is_bool_dtype(series.dtype):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    unexpected = set(normalized.unique()) - {"true", "false"}
    if unexpected:
        raise ValueError(f"Unexpected boolean values: {sorted(unexpected)}")
    return normalized.eq("true")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if total < 1:
        raise ValueError("total must be positive.")
    if not 0 <= successes <= total:
        raise ValueError("successes must lie between zero and total.")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_retrieval(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize top-1, candidate preferences, pairwise wins, and margins."""

    required = [
        "model",
        "seed",
        "predicted_label",
        "top1_correct",
        "true_margin",
        *[f"score_{name}" for name in CANDIDATE_NAMES],
    ]
    _require_columns(frame, required)

    rows: list[dict[str, float | int | str]] = []
    for (model, seed), group in frame.groupby(["model", "seed"], sort=False):
        correct = _boolean_values(group["top1_correct"])
        total = len(group)
        successes = int(correct.sum())
        lower, upper = wilson_interval(successes, total)
        row: dict[str, float | int | str] = {
            "model": str(model),
            "seed": int(seed),
            "examples": total,
            "top1_correct": successes,
            "top1_accuracy": successes / total,
            "top1_wilson_low": lower,
            "top1_wilson_high": upper,
            "mean_true_margin": float(group["true_margin"].mean()),
            "median_true_margin": float(group["true_margin"].median()),
        }
        predicted_rates = group["predicted_label"].value_counts(normalize=True)
        for candidate in CANDIDATE_NAMES:
            row[f"predicted_{candidate}_rate"] = float(
                predicted_rates.get(candidate, 0.0)
            )
        for candidate in CANDIDATE_NAMES[1:]:
            gap = group[f"score_{candidate}"] - group["score_true"]
            row[f"true_beats_{candidate}_rate"] = float((gap > 0).mean())
            row[f"mean_gap_{candidate}_minus_true"] = float(gap.mean())
            row[f"median_gap_{candidate}_minus_true"] = float(gap.median())
        rows.append(row)
    return pd.DataFrame(rows)


def merge_test_metadata(
    retrieval: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Attach test transition metadata to every model's retrieval row."""

    _require_columns(retrieval, ["sample_id", "model", "seed", "top1_correct"])
    _require_columns(
        metadata,
        [
            "split",
            "sample_id",
            "crowding",
            "stroke_width",
            "stroke_value",
            "stroke_length",
        ],
    )
    test_metadata = metadata.loc[metadata["split"] == "test"].copy()
    if test_metadata["sample_id"].duplicated().any():
        raise ValueError("Test metadata contains duplicate sample_id values.")
    return retrieval.merge(
        test_metadata[
            [
                "sample_id",
                "crowding",
                "stroke_width",
                "stroke_value",
                "stroke_length",
            ]
        ],
        on="sample_id",
        how="left",
        validate="many_to_one",
    )


def summarize_retrieval_by(
    merged: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Summarize retrieval by one test-transition property."""

    _require_columns(
        merged,
        [
            "model",
            "seed",
            group_column,
            "top1_correct",
            "true_margin",
            *[f"score_{name}" for name in CANDIDATE_NAMES],
        ],
    )
    rows: list[dict[str, float | int | str]] = []
    for (model, seed, group_value), group in merged.groupby(
        ["model", "seed", group_column],
        sort=False,
        observed=True,
    ):
        correct = _boolean_values(group["top1_correct"])
        total = len(group)
        successes = int(correct.sum())
        lower, upper = wilson_interval(successes, total)
        row: dict[str, float | int | str] = {
            "model": str(model),
            "seed": int(seed),
            group_column: group_value,
            "examples": total,
            "top1_correct": successes,
            "top1_accuracy": successes / total,
            "top1_wilson_low": lower,
            "top1_wilson_high": upper,
            "mean_true_margin": float(group["true_margin"].mean()),
            "median_true_margin": float(group["true_margin"].median()),
        }
        for candidate in CANDIDATE_NAMES[1:]:
            gap = group[f"score_{candidate}"] - group["score_true"]
            row[f"true_beats_{candidate}_rate"] = float((gap > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)
