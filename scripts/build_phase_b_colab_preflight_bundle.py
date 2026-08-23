#!/usr/bin/env python3
"""Package the exact branch and six frozen resources for Colab preflight."""

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

from latent_stroke_dynamics.phase_b_cloud_preflight import (  # noqa: E402
    FROZEN_RESOURCE_RAW_HASHES,
    raw_file_sha256,
    validate_cloud_preflight_boundary,
    verify_raw_resources,
)


BRANCH = "phase-b/saliency-latent"


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True
    ).strip()


def main() -> None:
    validate_cloud_preflight_boundary(ROOT)
    verify_raw_resources(ROOT)
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
    output = output_dir / f"phase-b0-colab-preflight-{head[:12]}.tar.gz"
    if output.exists():
        raise FileExistsError(f"Preflight bundle already exists: {output}")

    with tempfile.TemporaryDirectory(prefix="phase-b0-colab-") as temporary:
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
            "status": "phase_b0_colab_preflight_bundle_recovery_unauthorized",
            "branch": BRANCH,
            "source_commit": head,
            "resources": FROZEN_RESOURCE_RAW_HASHES,
            "contains_local_incomplete_output": False,
            "contains_scientific_phase_b_output": False,
            "recovery_authorized": False,
        }
        (staging / "bundle_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        with tarfile.open(output, "w:gz") as archive:
            archive.add(repository_bundle, arcname="repository.bundle")
            archive.add(staging / "bundle_manifest.json", arcname="bundle_manifest.json")
            archive.add(resources_root, arcname="resources")

    result = {
        "status": "phase_b0_colab_preflight_bundle_created",
        "path": str(output.relative_to(ROOT)),
        "sha256": raw_file_sha256(output),
        "size_bytes": output.stat().st_size,
        "source_commit": head,
        "resource_count": len(FROZEN_RESOURCE_RAW_HASHES),
        "recovery_authorized": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
