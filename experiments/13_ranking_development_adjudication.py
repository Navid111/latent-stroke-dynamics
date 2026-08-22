#!/usr/bin/env python3
"""Adjudicate completed ranking development artifacts without rerunning models."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from latent_stroke_dynamics.ranking_development_adjudication import (
    adjudicate_development,
)


DEFAULT_DIR = Path("outputs/ranking-aware-latent-development-2026-08-22")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args()


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite adjudication: {path}")
    temporary = path.with_suffix(path.suffix + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"Preserved incomplete adjudication exists: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    paths = {
        "summary": args.input_dir / "development_summary.json",
        "metrics": args.input_dir / "prediction_metrics.csv",
        "retrieval": args.input_dir / "counterfactual_retrieval.csv",
        "history": args.input_dir / "training_history.csv",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} artifact: {path}")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    metrics = pd.read_csv(paths["metrics"])
    retrieval = pd.read_csv(paths["retrieval"])
    history = pd.read_csv(paths["history"])
    result = adjudicate_development(summary, metrics, retrieval, history)
    result["source_artifact_sha256"] = {
        name: digest(path) for name, path in paths.items()
    }
    result["source_directory"] = str(args.input_dir)
    output = args.input_dir / "development_protocol_adjudication.json"
    write_new_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nSaved no-rerun adjudication to: {output.resolve()}")


if __name__ == "__main__":
    main()
