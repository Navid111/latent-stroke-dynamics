"""Read-only audit of Phase B0 transition-manifest fingerprint overlap."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from itertools import combinations
from pathlib import Path
from typing import Any


MANIFEST_FILES = {
    "train": "train_transitions.json",
    "validation": "validation_transitions.json",
    "diagnostic_test": "diagnostic_test_transitions.json",
}
EXPECTED_CLOUD_NATIVE_TRANSITION_MANIFEST_SHA256 = {
    "train_transitions.json": (
        "18551716942c747ee3daf8728bf1a8d1d21b9b075f85d71fe1365bcfd6a6e6e8"
    ),
    "validation_transitions.json": (
        "2b4fe2b782699538b91d3d13b453051fdb7e957d55fd371aba1cfdf56b44600a"
    ),
    "diagnostic_test_transitions.json": (
        "97d7e6527b27ade5671732fd025e069cb4497c85e64ad6c853c0a3cf0cbfee0b"
    ),
}
_HEX = frozenset("0123456789abcdef")


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blank_no_op_fingerprint(canvas_size: int = 64) -> str:
    """Derive the exact fingerprint of a white canvas plus a zero no-op action."""

    if canvas_size != 64:
        raise ValueError("The frozen Phase B0 audit requires a 64x64 canvas.")
    pixels = canvas_size * canvas_size
    digest = sha256()
    white_canvas = bytes([255]) * pixels
    digest.update(white_canvas)
    digest.update(white_canvas)
    # Phase B0 actions contain two 64x64 float32 channels. All-zero float32
    # values are byte-identical across the audited Linux and macOS platforms.
    digest.update(bytes(2 * pixels * 4))
    digest.update(b"\x01")
    return digest.hexdigest()


def _load_manifest(path: Path, expected_split: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    if payload.get("split") != expected_split:
        raise ValueError(
            f"Manifest split mismatch in {path}: expected {expected_split!r}."
        )
    samples = payload.get("samples")
    no_op_samples = payload.get("no_op_samples")
    fingerprints = payload.get("fingerprints")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError(f"Invalid sample count in {path}.")
    if (
        isinstance(no_op_samples, bool)
        or not isinstance(no_op_samples, int)
        or not 0 <= no_op_samples <= samples
    ):
        raise ValueError(f"Invalid no-op count in {path}.")
    if not isinstance(fingerprints, list) or len(fingerprints) != samples:
        raise ValueError(f"Fingerprint count does not match samples in {path}.")
    for value in fingerprints:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in _HEX for character in value)
        ):
            raise ValueError(f"Invalid SHA-256 fingerprint in {path}.")
    return payload


def audit_transition_manifest_overlap(
    manifest_dir: str | Path,
    *,
    require_cloud_native_hashes: bool = True,
) -> dict[str, Any]:
    """Inspect frozen manifests without regenerating data or loading any model."""

    root = Path(manifest_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Manifest directory does not exist: {root}")
    paths = {name: root / filename for name, filename in MANIFEST_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required transition manifests: {missing}")

    hashes_before = {path.name: file_sha256(path) for path in paths.values()}
    cloud_native_hash_gate = (
        hashes_before == EXPECTED_CLOUD_NATIVE_TRANSITION_MANIFEST_SHA256
    )
    if require_cloud_native_hashes and not cloud_native_hash_gate:
        mismatches = {
            name: {
                "expected": EXPECTED_CLOUD_NATIVE_TRANSITION_MANIFEST_SHA256[name],
                "actual": hashes_before.get(name),
            }
            for name in EXPECTED_CLOUD_NATIVE_TRANSITION_MANIFEST_SHA256
            if hashes_before.get(name)
            != EXPECTED_CLOUD_NATIVE_TRANSITION_MANIFEST_SHA256[name]
        }
        raise ValueError(
            "The supplied files are not the canonical completed cloud-native "
            f"transition manifests. Hash mismatches: {mismatches}"
        )

    manifests = {
        name: _load_manifest(path, name) for name, path in paths.items()
    }
    hashes_after = {path.name: file_sha256(path) for path in paths.values()}
    if hashes_before != hashes_after:
        raise RuntimeError("A source transition manifest changed during the audit.")

    blank = blank_no_op_fingerprint()
    fingerprint_lists = {
        name: tuple(str(value) for value in payload["fingerprints"])
        for name, payload in manifests.items()
    }
    fingerprint_sets = {
        name: set(values) for name, values in fingerprint_lists.items()
    }

    split_summaries: dict[str, Any] = {}
    for name, values in fingerprint_lists.items():
        counts = Counter(values)
        split_summaries[name] = {
            "samples": int(manifests[name]["samples"]),
            "no_op_samples": int(manifests[name]["no_op_samples"]),
            "unique_fingerprints": len(counts),
            "duplicate_fingerprint_kinds": sum(
                1 for count in counts.values() if count > 1
            ),
            "duplicate_entries_beyond_first": sum(
                count - 1 for count in counts.values() if count > 1
            ),
            "maximum_fingerprint_multiplicity": max(counts.values()),
            "blank_no_op_fingerprint_multiplicity": counts.get(blank, 0),
        }

    pairwise: dict[str, Any] = {}
    overlap_union: set[str] = set()
    for left, right in combinations(MANIFEST_FILES, 2):
        overlap = fingerprint_sets[left] & fingerprint_sets[right]
        overlap_union.update(overlap)
        pairwise[f"{left}__{right}"] = {
            "count": len(overlap),
            "fingerprints": sorted(overlap),
            "blank_no_op_only": bool(overlap) and overlap <= {blank},
        }

    triple = set.intersection(*(fingerprint_sets[name] for name in MANIFEST_FILES))
    nonblank_overlap = overlap_union - {blank}
    if not overlap_union:
        classification = "no_cross_split_overlap"
        interpretation = "The three transition manifests are strictly disjoint."
    elif not nonblank_overlap:
        classification = "blank_no_op_only"
        interpretation = (
            "Every cross-split collision is the analytically derived all-white "
            "no-op transition; no changing or nonblank transition fingerprint "
            "is shared across splits."
        )
    else:
        classification = "contains_nonblank_or_unknown_overlap"
        interpretation = (
            "At least one shared fingerprint is not the analytically derived "
            "all-white no-op transition and requires further investigation."
        )

    return {
        "status": "phase_b0_transition_overlap_audit_complete",
        "classification": classification,
        "interpretation": interpretation,
        "manifest_directory": str(root),
        "expected_cloud_native_transition_manifest_sha256": (
            EXPECTED_CLOUD_NATIVE_TRANSITION_MANIFEST_SHA256
        ),
        "source_manifest_sha256": hashes_before,
        "cloud_native_manifest_hash_gate_passed": cloud_native_hash_gate,
        "source_files_unchanged": hashes_before == hashes_after,
        "expected_blank_no_op_fingerprint": blank,
        "split_summaries": split_summaries,
        "pairwise_intersections": pairwise,
        "triple_intersection": {
            "count": len(triple),
            "fingerprints": sorted(triple),
            "blank_no_op_only": bool(triple) and triple <= {blank},
        },
        "cross_split_overlap_union_count": len(overlap_union),
        "cross_split_overlap_fingerprints": sorted(overlap_union),
        "strict_transition_splits_disjoint": not overlap_union,
        "nonblank_transition_splits_disjoint": not nonblank_overlap,
        "all_cross_split_overlaps_are_blank_no_op": (
            bool(overlap_union) and not nonblank_overlap
        ),
        "nonblank_or_unknown_overlap_count": len(nonblank_overlap),
        "nonblank_or_unknown_overlap_fingerprints": sorted(nonblank_overlap),
        "scientific_side_effects": {
            "renderer_data_generated": False,
            "models_loaded": False,
            "models_trained": False,
            "checkpoints_written": False,
            "historical_output_modified": False,
        },
        "decision_context": {
            "original_implementation_integrity_result_changed": False,
            "original_cloud_native_decision_changed": False,
            "rerun_authorized": False,
            "formal_phase_authorized": False,
            "phase_b1_authorized": False,
            "phase_b2_authorized": False,
        },
    }
