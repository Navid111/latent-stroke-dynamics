import json
from hashlib import sha256
from pathlib import Path

from PIL import Image
import pytest
import torch

import latent_stroke_dynamics.phase_b_qualitative_painter as qualitative
from latent_stroke_dynamics.extension_training import model_state_sha256
from latent_stroke_dynamics.phase_b_joint_embedding import (
    EXPECTED_TRAINABLE_PARAMETERS,
    MultiScaleActionJointEmbeddingModel,
)
from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke


def _sha(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkpoint(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    variant: str = "joint_prediction_only",
) -> tuple[Path, str]:
    torch.manual_seed(73)
    model = MultiScaleActionJointEmbeddingModel()
    state_digest = model_state_sha256(model)
    torch.save(
        {
            "format_version": 1,
            "architecture": "MultiScaleActionJointEmbeddingModel",
            "variant": variant,
            "seed": 73,
            "best_epoch": 40,
            "best_validation_loss": 0.006895331389387138,
            "training_device": "cuda:0",
            "trainable_parameter_count": EXPECTED_TRAINABLE_PARAMETERS,
            "progress_training_mean": 0.001,
            "progress_training_std": 0.01,
            "state_sha256": state_digest,
            "state_dict": model.state_dict(),
        },
        path,
    )
    monkeypatch.setattr(
        qualitative,
        "EXPECTED_PREDICTION_ONLY_ARTIFACT_SHA256",
        _sha(path),
    )
    monkeypatch.setattr(
        qualitative,
        "EXPECTED_PREDICTION_ONLY_STATE_SHA256",
        state_digest,
    )
    return path, state_digest


def test_prediction_only_checkpoint_loader_verifies_and_freezes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, expected_state = _checkpoint(tmp_path / "prediction.pt", monkeypatch)

    model, metadata = qualitative.load_prediction_only_checkpoint(checkpoint)

    assert metadata["variant"] == "joint_prediction_only"
    assert metadata["state_sha256"] == expected_state
    assert metadata["artifact_sha256"] == _sha(checkpoint)
    assert not any(parameter.requires_grad for parameter in model.parameters())


def test_prediction_only_checkpoint_loader_rejects_other_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _ = _checkpoint(
        tmp_path / "progress.pt",
        monkeypatch,
        variant="joint_prediction_progress",
    )

    with pytest.raises(ValueError, match="variant"):
        qualitative.load_prediction_only_checkpoint(checkpoint)


def test_qualitative_latent_comparison_is_inference_only_and_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint, _ = _checkpoint(tmp_path / "prediction.pt", monkeypatch)
    target = tmp_path / "target.png"
    render_stroke(
        blank_canvas(64),
        Stroke(0.15, 0.2, 0.85, 0.8, width=3, value=0),
    ).save(target)
    checkpoint_before = checkpoint.read_bytes()
    target_before = target.read_bytes()
    output = tmp_path / "qualitative-output"

    completed, summary = qualitative.run_prediction_only_qualitative_comparison(
        target,
        checkpoint,
        output,
        polarity="preserve",
        steps=2,
        candidates=4,
        seed=123,
        prediction_batch_size=2,
        device="cpu",
        high_res_size=64,
        supersample=1,
    )

    assert completed == output
    assert summary["status"] == "phase_b0_prediction_only_qualitative_inference_complete"
    assert summary["models_trained"] is False
    assert summary["formal_claims_allowed"] is False
    assert summary["source_phase_b0_decision"] == "not_eligible"
    assert summary["latent"]["executed_steps"] == 2
    assert summary["exact_pixel"]["executed_steps"] == 2
    assert checkpoint.read_bytes() == checkpoint_before
    assert target.read_bytes() == target_before
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert config["training_performed"] is False
    assert config["rerun_of_completed_training"] is False
    assert config["source_decision_preserved"] == "not_eligible"
    assert (output / "comparison.png").is_file()
    assert (output / "progress_comparison.png").is_file()
    assert (output / "latent_prediction_only" / "best_64.png").is_file()
    assert (output / "latent_prediction_only" / "painting_64.gif").is_file()
    assert (output / "exact_pixel" / "best_64.png").is_file()
    with Image.open(output / "latent_prediction_only" / "best_64.png") as image:
        assert image.size == (64, 64)

    with pytest.raises(FileExistsError, match="overwrite"):
        qualitative.run_prediction_only_qualitative_comparison(
            target,
            checkpoint,
            output,
            polarity="preserve",
            steps=1,
            candidates=2,
            prediction_batch_size=1,
            device="cpu",
            high_res_size=64,
            supersample=1,
        )
