#!/usr/bin/env python3
"""Adjudicate the completed extension summary without data/model execution."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

from latent_stroke_dynamics.extension_adjudication import adjudicate_extension


DEFAULT_INPUT = Path(
    "outputs/representation-extension-2026-08-22/extension_summary.json"
)
DEFAULT_OUTPUT = Path(
    "outputs/representation-extension-2026-08-22/protocol_adjudication.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete artifact: {temporary}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    source_bytes = args.input.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    result = adjudicate_extension(source)
    result["source_summary"] = str(args.input)
    result["source_summary_sha256"] = sha256(source_bytes).hexdigest()
    write_new_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nSaved no-rerun adjudication to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
