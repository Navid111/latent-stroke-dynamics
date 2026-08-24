#!/usr/bin/env python3
"""Package the separately authorized cloud-native Phase B0 execution."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latent_stroke_dynamics.phase_b_cloud_native import (  # noqa: E402
    DEFAULT_CLOUD_NATIVE_CONFIG,
    load_cloud_native_execution_config,
)
from latent_stroke_dynamics.phase_b_cloud_preflight import (  # noqa: E402
    FROZEN_RESOURCE_RAW_HASHES,
    raw_file_sha256,
    verify_raw_resources,
)


BRANCH = "phase-b/saliency-latent"
EXPECTED_TEST_COUNT = 160


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def main() -> None:
    config = load_cloud_native_execution_config(ROOT / DEFAULT_CLOUD_NATIVE_CONFIG)
    authorization = config["cloud_native_authorization"]
    verify_raw_resources(ROOT)
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError(f"Expected branch {BRANCH!r}.")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked working-tree changes are present.")
    head = git("rev-parse", "HEAD")
    parent = git("rev-parse", "HEAD^")
    if parent != authorization["validated_handoff_commit"]:
        raise RuntimeError(
            "Authorization must be the direct child of the validated handoff commit."
        )
    output_dir = ROOT / "dist"
    output_dir.mkdir(exist_ok=True)
    output = output_dir / f"phase-b0-colab-native-execution-{head[:12]}.tar.gz"
    if output.exists():
        raise FileExistsError(f"Cloud-native execution bundle exists: {output}")

    with tempfile.TemporaryDirectory(prefix="phase-b0-cloud-native-") as temporary:
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
        resources_root = staging / "resources"
        for relative in FROZEN_RESOURCE_RAW_HASHES:
            destination = resources_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        manifest = {
            "status": "phase_b0_colab_native_execution_bundle_authorized_once",
            "experiment_id": config["experiment_id"],
            "branch": BRANCH,
            "source_commit": head,
            "authorized_parent_commit": parent,
            "authorization": authorization,
            "expected_test_count": EXPECTED_TEST_COUNT,
            "artifact_root": authorization["external_artifact_root"],
            "resources": FROZEN_RESOURCE_RAW_HASHES,
            "resource_count": len(FROZEN_RESOURCE_RAW_HASHES),
            "new_experiment": True,
            "recovery_of_mac_attempt": False,
            "maximum_completed_executions": 1,
            "cloud_native_development_authorized": True,
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
            archive.add(resources_root, arcname="resources")

    print(
        json.dumps(
            {
                "status": "phase_b0_colab_native_execution_bundle_created",
                "path": str(output.relative_to(ROOT)),
                "sha256": raw_file_sha256(output),
                "size_bytes": output.stat().st_size,
                "source_commit": head,
                "resource_count": len(FROZEN_RESOURCE_RAW_HASHES),
                "expected_test_count": EXPECTED_TEST_COUNT,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
