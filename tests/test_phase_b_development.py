from copy import deepcopy
import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from latent_stroke_dynamics.phase_b_data import (
    PlannerCandidateSet,
    PlannerTensorPayload,
    TransitionTensorPayload,
)
from latent_stroke_dynamics.phase_b_development import (
    AUTHORIZED_STATUS,
    LONG_HORIZON_METHODS,
    PhaseBOutputPaths,
    load_phase_b_development_config,
    phase_b_output_paths,
    require_phase_b_development_authorized,
    require_phase_b_outputs_absent,
    validate_phase_b_development_runner_request,
)
from latent_stroke_dynamics.phase_b_training import train_phase_b_variant


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phase-b-saliency-latent-2026-08-23.json"


def test_phase_b_development_runner_is_valid_and_unauthorized() -> None:
    config = load_phase_b_development_config(CONFIG)
    paths = phase_b_output_paths(config)
    assert not paths.final.exists()
    assert not paths.incomplete.exists()
    result = validate_phase_b_development_runner_request(config)
    assert result["status"] == "phase_b0_development_runner_valid_unauthorized"
    assert result["variants"] == ["joint_prediction_only", "joint_prediction_progress"]
    assert result["transition_samples"] == {
        "train": 2048,
        "validation": 512,
        "diagnostic_test": 512,
    }
    assert result["planner_training_candidate_sets"] == 64
    assert result["planner_validation_candidate_sets"] == 32
    assert tuple(result["long_horizon_methods"]) == LONG_HORIZON_METHODS
    assert result["historical_models_loaded"] is False
    assert result["renderer_transitions_generated"] is False
    assert result["output_directories_created"] is False
    assert not paths.final.exists()
    assert not paths.incomplete.exists()


def test_phase_b_development_guard_stops_before_any_side_effect() -> None:
    config = load_phase_b_development_config(CONFIG)
    paths = phase_b_output_paths(config)
    with pytest.raises(PermissionError, match="not authorized"):
        require_phase_b_development_authorized(config)
    assert not paths.final.exists()
    assert not paths.incomplete.exists()


def test_phase_b_authorized_lifecycle_copy_preserves_static_protocol(tmp_path: Path) -> None:
    config = load_phase_b_development_config(CONFIG)
    authorized = deepcopy(config)
    authorized["status"] = AUTHORIZED_STATUS
    authorized["development"]["authorized"] = True
    path = tmp_path / "authorized.json"
    path.write_text(json.dumps(authorized), encoding="utf-8")
    loaded = load_phase_b_development_config(path)
    assert loaded["status"] == AUTHORIZED_STATUS
    assert loaded["development"]["authorized"] is True
    broken = deepcopy(authorized)
    broken["development"]["authorized"] = False
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        load_phase_b_development_config(path)


def test_phase_b_output_guard_preserves_completed_and_incomplete_runs(tmp_path: Path) -> None:
    final = tmp_path / "development"
    paths = PhaseBOutputPaths(final=final, incomplete=tmp_path / "development.incomplete")
    require_phase_b_outputs_absent(paths)
    final.mkdir()
    with pytest.raises(FileExistsError, match="output exists"):
        require_phase_b_outputs_absent(paths)
    final.rmdir()
    paths.incomplete.mkdir()
    with pytest.raises(FileExistsError, match="Incomplete"):
        require_phase_b_outputs_absent(paths)


def _dummy_payloads() -> tuple[
    TransitionTensorPayload,
    TransitionTensorPayload,
    PlannerTensorPayload,
]:
    generator = torch.Generator().manual_seed(700)
    current = torch.rand(2, 1, 64, 64, generator=generator)
    next_canvas = torch.rand(2, 1, 64, 64, generator=generator)
    actions = torch.rand(2, 2, 64, 64, generator=generator)
    actions[0].zero_()
    transition = TransitionTensorPayload(
        examples=(),
        current=current,
        next_canvas=next_canvas,
        actions=actions,
        no_op=torch.tensor([True, False]),
    )
    candidate_count = 32
    planner_current = torch.rand(candidate_count, 1, 64, 64, generator=generator)
    planner_next = torch.rand(candidate_count, 1, 64, 64, generator=generator)
    planner_target = torch.rand(candidate_count, 1, 64, 64, generator=generator)
    planner_actions = torch.rand(candidate_count, 2, 64, 64, generator=generator)
    planner_actions[0].zero_()
    exact_progress = torch.linspace(-0.02, 0.03, candidate_count)
    exact_progress[0] = 0.0
    blank = Image.new("L", (64, 64), 255)
    record = PlannerCandidateSet(
        set_id=0,
        target_seed=1,
        trajectory_seed=2,
        candidate_seed=3,
        state_name="blank",
        current=blank,
        target=blank,
        candidates=(None,) * candidate_count,
    )
    planner = PlannerTensorPayload(
        records=(record,),
        current=planner_current,
        next_canvas=planner_next,
        target=planner_target,
        actions=planner_actions,
        exact_progress=exact_progress,
        set_index=torch.zeros(candidate_count, dtype=torch.int64),
        candidate_index=torch.arange(candidate_count, dtype=torch.int64),
    )
    return transition, transition, planner


def test_phase_b_training_paths_reduce_to_finite_dummy_objectives() -> None:
    train, validation, planner = _dummy_payloads()
    for variant in ("joint_prediction_only", "joint_prediction_progress"):
        fit = train_phase_b_variant(
            variant,
            train,
            validation,
            planner,
            planner,
            progress_mean=0.0,
            progress_std=0.02,
            seed=73,
            learning_rate=0.0003,
            weight_decay=0.0001,
            batch_size=2,
            maximum_epochs=1,
            patience=1,
            gradient_clip_norm=5.0,
            wall_clock_cap_hours=1.0,
        )
        assert fit.best_epoch == 1
        assert fit.best_validation_loss < float("inf")
        assert len(fit.history) == 1
