import numpy as np
import torch

from latent_stroke_dynamics.gate2 import ACTION_DIM, parameter_count
from latent_stroke_dynamics.learned_pixel_planner import (
    CHECKPOINT_FORMAT_VERSION,
    CHECKPOINT_TYPE,
    PixelCheckpointMetadata,
    learned_candidate_scores,
    load_pixel_checkpoint,
    run_learned_planner,
    save_pixel_checkpoint,
)
from latent_stroke_dynamics.pixel_control import (
    PIXEL_INPUT_DIM,
    ExactCompositorPixelDeltaPredictor,
    MLPPixelDeltaPredictor,
)
from latent_stroke_dynamics.planning import ProposalConfig
from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke


def metadata_for(model: MLPPixelDeltaPredictor) -> PixelCheckpointMetadata:
    return PixelCheckpointMetadata(
        checkpoint_type=CHECKPOINT_TYPE,
        format_version=CHECKPOINT_FORMAT_VERSION,
        canvas_size=32,
        pixel_input_dim=PIXEL_INPUT_DIM,
        action_dim=ACTION_DIM,
        architecture="MLPPixelDeltaPredictor",
        hidden_dim=16,
        parameter_count=parameter_count(model),
        model_seed=11,
        train_seed=101,
        validation_seed=102,
        train_samples=8,
        validation_samples=4,
        crowding=(0, 2),
        epochs=2,
        patience=1,
        learning_rate=0.001,
        weight_decay=0.0001,
        batch_size=2,
        best_epoch=2,
        best_validation_loss=0.01,
        test_rows_used_for_training_or_selection=False,
    )


def test_checkpoint_round_trip_preserves_metadata_and_predictions(tmp_path) -> None:
    torch.manual_seed(11)
    model = MLPPixelDeltaPredictor(hidden_dim=16).eval()
    metadata = metadata_for(model)
    path = save_pixel_checkpoint(model, metadata, tmp_path / "model.pt")
    loaded, loaded_metadata = load_pixel_checkpoint(path)
    current = torch.rand(2, 32, 32)
    actions = torch.rand(2, ACTION_DIM)
    masks = torch.zeros(2, 32, 32)
    masks[:, 10:20, 5:25] = 1.0
    with torch.inference_mode():
        expected = model(current, actions, masks)
        actual = loaded(current, actions, masks)
    assert loaded_metadata == metadata
    assert torch.equal(expected, actual)


def test_exact_oracle_candidate_scores_retrieve_true_target() -> None:
    current = blank_canvas(32)
    true_stroke = Stroke(0.1, 0.2, 0.9, 0.8, width=3, value=32)
    wrong_stroke = Stroke(0.1, 0.8, 0.9, 0.2, width=1, value=128)
    target = render_stroke(current, true_stroke)
    scores = learned_candidate_scores(
        ExactCompositorPixelDeltaPredictor(),
        current,
        target,
        (wrong_stroke, true_stroke),
        batch_size=1,
    )
    assert int(np.argmin(scores)) == 1
    assert scores[1] == 0.0


def test_oracle_learned_loop_matches_exact_candidate_ranking() -> None:
    target = render_stroke(
        blank_canvas(32),
        Stroke(0.05, 0.2, 0.95, 0.8, width=4, value=0),
    )
    run = run_learned_planner(
        target,
        ExactCompositorPixelDeltaPredictor(),
        steps=2,
        seed=91,
        proposal_config=ProposalConfig(count=8, min_length=0.1, max_length=0.4),
        prediction_batch_size=4,
        capture_frames=True,
    )
    assert len(run.steps) == 2
    assert len(run.frames) == 3
    assert all(record.exact_selected_rank == 1 for record in run.steps)
    assert all(record.exact_top1 for record in run.steps)
    assert all(record.exact_regret <= 1e-12 for record in run.steps)
