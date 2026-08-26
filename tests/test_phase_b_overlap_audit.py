import json
from pathlib import Path

import pytest

from latent_stroke_dynamics.phase_b_overlap_audit import (
    audit_transition_manifest_overlap,
    blank_no_op_fingerprint,
)


def _write_manifest(
    root: Path,
    split: str,
    fingerprints: list[str],
    no_op_samples: int,
) -> Path:
    path = root / f"{split}_transitions.json"
    path.write_text(
        json.dumps(
            {
                "split": split,
                "seed": 1,
                "samples": len(fingerprints),
                "no_op_samples": no_op_samples,
                "fingerprints": fingerprints,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_overlap_audit_identifies_only_the_blank_no_op(tmp_path: Path) -> None:
    blank = blank_no_op_fingerprint()
    _write_manifest(tmp_path, "train", [blank, "a" * 64], 1)
    _write_manifest(tmp_path, "validation", [blank, "b" * 64], 1)
    _write_manifest(tmp_path, "diagnostic_test", [blank, "c" * 64], 1)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    report = audit_transition_manifest_overlap(
        tmp_path,
        require_cloud_native_hashes=False,
    )

    assert report["classification"] == "blank_no_op_only"
    assert report["strict_transition_splits_disjoint"] is False
    assert report["nonblank_transition_splits_disjoint"] is True
    assert report["cross_split_overlap_union_count"] == 1
    assert report["triple_intersection"]["fingerprints"] == [blank]
    assert report["source_files_unchanged"] is True
    assert report["cloud_native_manifest_hash_gate_passed"] is False
    assert before == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert not any(report["scientific_side_effects"].values())


def test_overlap_audit_flags_a_nonblank_shared_fingerprint(tmp_path: Path) -> None:
    blank = blank_no_op_fingerprint()
    shared = "d" * 64
    _write_manifest(tmp_path, "train", [blank, shared], 1)
    _write_manifest(tmp_path, "validation", [blank, shared], 1)
    _write_manifest(tmp_path, "diagnostic_test", [blank, "e" * 64], 1)

    report = audit_transition_manifest_overlap(
        tmp_path,
        require_cloud_native_hashes=False,
    )

    assert report["classification"] == "contains_nonblank_or_unknown_overlap"
    assert report["nonblank_transition_splits_disjoint"] is False
    assert report["nonblank_or_unknown_overlap_fingerprints"] == [shared]


def test_overlap_audit_rejects_manifest_sample_mismatch(tmp_path: Path) -> None:
    blank = blank_no_op_fingerprint()
    path = _write_manifest(tmp_path, "train", [blank], 1)
    _write_manifest(tmp_path, "validation", ["b" * 64], 0)
    _write_manifest(tmp_path, "diagnostic_test", ["c" * 64], 0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Fingerprint count"):
        audit_transition_manifest_overlap(
            tmp_path,
            require_cloud_native_hashes=False,
        )


def test_overlap_audit_rejects_noncanonical_manifests_by_default(
    tmp_path: Path,
) -> None:
    blank = blank_no_op_fingerprint()
    _write_manifest(tmp_path, "train", [blank], 1)
    _write_manifest(tmp_path, "validation", [blank], 1)
    _write_manifest(tmp_path, "diagnostic_test", [blank], 1)

    with pytest.raises(ValueError, match="canonical completed cloud-native"):
        audit_transition_manifest_overlap(tmp_path)
