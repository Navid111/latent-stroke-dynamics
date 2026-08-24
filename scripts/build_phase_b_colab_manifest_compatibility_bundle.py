#!/usr/bin/env python3
"""Package the manifest-only compatibility gate without model resources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_stroke_dynamics.phase_b_manifest_compatibility import (  # noqa: E402
    EXPECTED_ORIGINAL_MANIFEST_HASHES,
    PASS_STATUS,
    file_sha256,
    load_manifest_compatibility_config,
)


BRANCH = "phase-b/saliency-latent"
EXPECTED_LOCAL_TEST_COUNT = 153
TARGETED_COLAB_TEST_COUNT = 8


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-report", type=Path, required=True)
    parser.add_argument("--local-test-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_manifest_compatibility_config()
    if args.local_test_count != EXPECTED_LOCAL_TEST_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LOCAL_TEST_COUNT} passing local tests, received {args.local_test_count}."
        )
    report_path = args.local_report.expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != PASS_STATUS or report.get("hash_gate_passed") is not True:
        raise RuntimeError("Local manifest-only validation did not pass all four hashes.")
    if report.get("scientific_models_trained") is not False:
        raise RuntimeError("Local compatibility report crossed the non-training boundary.")
    if report.get("recovery_authorized") is not False:
        raise RuntimeError("Local compatibility report unexpectedly authorized recovery.")

    branch = git("branch", "--show-current")
    if branch != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH!r}, found {branch!r}.")
    tracked_changes = git("status", "--porcelain", "--untracked-files=no")
    if tracked_changes:
        raise RuntimeError(
            "Tracked working tree changes are present. Commit or restore them before packaging."
        )
    head = git("rev-parse", "HEAD")
    output_dir = ROOT / "dist"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / f"phase-b0-colab-manifest-compatibility-{head[:12]}.tar.gz"
    if output.exists():
        raise FileExistsError(f"Manifest-compatibility bundle already exists: {output}")

    with tempfile.TemporaryDirectory(prefix="phase-b0-manifest-compatibility-") as temporary:
        staging = Path(temporary)
        repository_bundle = staging / "repository.bundle"
        subprocess.run(
            ["git", "bundle", "create", str(repository_bundle), BRANCH],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            ["git", "bundle", "verify", str(repository_bundle)],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        manifest = {
            "status": "phase_b0_colab_manifest_compatibility_bundle_unauthorized",
            "purpose": "pinned_manifest_only_cross_platform_compatibility_check",
            "branch": BRANCH,
            "source_commit": head,
            "entrypoint": "experiments/27_phase_b_colab_manifest_compatibility.py",
            "requirements": "requirements/phase-b0-manifest-compatibility-2026-08-24.txt",
            "targeted_tests": "tests/test_phase_b_manifest_compatibility.py",
            "expected_local_test_count": EXPECTED_LOCAL_TEST_COUNT,
            "targeted_colab_test_count": TARGETED_COLAB_TEST_COUNT,
            "local_compatibility_report_sha256": file_sha256(report_path),
            "expected_original_manifest_sha256": EXPECTED_ORIGINAL_MANIFEST_HASHES,
            "required_environment": config["required_environment"],
            "resource_count": 0,
            "contains_model_resources": False,
            "contains_historical_incomplete_output": False,
            "contains_recovery_output": False,
            "scientific_training_allowed": False,
            "recovery_authorized": False,
            "formal_authorized": False,
            "phase_b1_authorized": False,
            "phase_b2_authorized": False,
        }
        (staging / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        with tarfile.open(output, "w:gz") as archive:
            archive.add(repository_bundle, arcname="repository.bundle")
            archive.add(staging / "bundle_manifest.json", arcname="bundle_manifest.json")

    result = {
        "status": "phase_b0_colab_manifest_compatibility_bundle_created_unauthorized",
        "path": str(output.relative_to(ROOT)),
        "sha256": file_sha256(output),
        "size_bytes": output.stat().st_size,
        "source_commit": head,
        "resource_count": 0,
        "expected_local_test_count": EXPECTED_LOCAL_TEST_COUNT,
        "targeted_colab_test_count": TARGETED_COLAB_TEST_COUNT,
        "scientific_training_allowed": False,
        "recovery_authorized": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
