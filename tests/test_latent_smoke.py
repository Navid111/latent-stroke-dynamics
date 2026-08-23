from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from latent_stroke_dynamics.extension_training import create_patch_predictor
from latent_stroke_dynamics.latent_planner import load_latent_planner_config
from latent_stroke_dynamics.latent_smoke import (
    require_smoke_authorized,
    require_smoke_outputs_absent,
    run_latent_planner,
    smoke_output_paths,
    spearman_rank_correlation,
    validate_smoke_runner_request,
)
from latent_stroke_dynamics.planning import ProposalConfig
from latent_stroke_dynamics.representation_extension import (
    LatentChannelStatistics,
    StrokeAutoencoder,
    freeze_module,
)
from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "latent-planner-2026-08-23.json"


def test_smoke_validation_is_unauthorized_and_side_effect_free(tmp_path: Path) -> None:
    config = deepcopy(load_latent_planner_config(CONFIG))
    config["smoke"]["output_dir"] = str(tmp_path / "smoke")
    result = validate_smoke_runner_request(config)
    assert result["status"] == "latent_planner_smoke_runner_valid_unauthorized"
    assert result["smoke_authorized"] is False
    assert result["models_loaded"] is False
    assert result["target_generated"] is False
    assert result["planner_data_generated"] is False
    assert not (tmp_path / "smoke").exists()
    assert not (tmp_path / "smoke.incomplete").exists()


def test_smoke_run_is_rejected_before_authorization() -> None:
    config = load_latent_planner_config(CONFIG)
    with pytest.raises(PermissionError, match="not authorized"):
        require_smoke_authorized(config)


def test_smoke_output_guard_preserves_existing_incomplete(tmp_path: Path) -> None:
    config = deepcopy(load_latent_planner_config(CONFIG))
    config["smoke"]["output_dir"] = str(tmp_path / "smoke")
    paths = smoke_output_paths(config)
    paths.incomplete.mkdir()
    marker = paths.incomplete / "preserve.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Incomplete"):
        require_smoke_outputs_absent(paths)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_spearman_rank_correlation_is_tie_aware_and_finite() -> None:
    assert spearman_rank_correlation(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([10.0, 20.0, 30.0]),
    ) == 1.0
    assert spearman_rank_correlation(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([30.0, 20.0, 10.0]),
    ) == -1.0
    tied = spearman_rank_correlation(
        np.asarray([1.0, 1.0, 2.0]),
        np.asarray([2.0, 2.0, 3.0]),
    )
    assert np.isfinite(tied)


def test_latent_planner_is_deterministic_and_executes_selected_stroke_exactly() -> None:
    torch.manual_seed(7)
    autoencoder = freeze_module(StrokeAutoencoder())
    statistics = LatentChannelStatistics(
        mean=torch.zeros(32),
        std=torch.ones(32),
    )
    predictors = []
    for seed in (11, 22, 33):
        torch.manual_seed(seed)
        predictors.append(
            freeze_module(create_patch_predictor("mlp", 32, (16, 16), 256))
        )
    target = render_stroke(
        blank_canvas(64),
        Stroke(0.1, 0.2, 0.9, 0.8, width=3, value=32),
    )
    proposal = ProposalConfig(
        count=4,
        error_guided_fraction=1.0,
        min_length=0.1,
        max_length=0.2,
    )
    first = run_latent_planner(
        target,
        "latent_ranking",
        autoencoder,
        statistics,
        predictors,
        steps=1,
        seed=515,
        proposal_config=proposal,
        prediction_batch_size=2,
        capture_frames=True,
    )
    second = run_latent_planner(
        target,
        "latent_ranking",
        autoencoder,
        statistics,
        predictors,
        steps=1,
        seed=515,
        proposal_config=proposal,
        prediction_batch_size=4,
        capture_frames=False,
    )
    assert first.steps == second.steps
    assert first.target_encoding_count == 1
    assert first.observed_canvas_encoding_count == 1
    assert len(first.frames) == 2
    exact = render_stroke(first.initial_canvas, first.steps[0].stroke)
    assert np.array_equal(np.asarray(first.final_canvas), np.asarray(exact))
    assert np.array_equal(np.asarray(first.final_canvas), np.asarray(second.final_canvas))
