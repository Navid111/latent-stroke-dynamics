import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from latent_stroke_dynamics.high_resolution_replay import (
    replay_existing_painting,
    replay_strokes_high_resolution,
)
from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke


def _source_painting(root: Path) -> Path:
    source = root / "painting"
    source.mkdir()
    target = render_stroke(
        blank_canvas(64),
        Stroke(0.1, 0.2, 0.9, 0.8, width=2, value=32),
    )
    target.save(source / "processed_target.png")
    strokes = [
        {
            "step": 1,
            "selected_index": 3,
            "x0": 0.1,
            "y0": 0.2,
            "x1": 0.9,
            "y1": 0.8,
            "width": 2,
            "value": 32,
        },
        {
            "step": 2,
            "selected_index": 1,
            "x0": 0.2,
            "y0": 0.8,
            "x1": 0.8,
            "y1": 0.2,
            "width": 1,
            "value": 64,
        },
    ]
    (source / "strokes.json").write_text(json.dumps(strokes), encoding="utf-8")
    (source / "summary.json").write_text(
        json.dumps({"best_step": 1}),
        encoding="utf-8",
    )
    (source / "run_config.json").write_text(
        json.dumps(
            {
                "qualitative_demo": True,
                "target_processing": {"canvas_size": 64},
            }
        ),
        encoding="utf-8",
    )
    (source / "progress.csv").write_text(
        "step,mse_before,mse_after\n"
        "1,0.400000,0.100000\n"
        "2,0.100000,0.200000\n",
        encoding="utf-8",
    )
    return source


def test_high_resolution_replay_scales_fixed_strokes() -> None:
    stroke = Stroke(0.1, 0.5, 0.9, 0.5, width=2, value=0)
    frames = replay_strokes_high_resolution(
        (stroke,),
        output_size=128,
        supersample=2,
    )

    assert len(frames) == 2
    assert all(frame.size == (128, 128) for frame in frames)
    assert np.asarray(frames[0]).min() == 255
    assert np.asarray(frames[1]).min() < 255


def test_existing_painting_replay_is_atomic_and_read_only(tmp_path: Path) -> None:
    source = _source_painting(tmp_path)
    output = tmp_path / "painting-128"
    source_before = {path.name: path.read_bytes() for path in source.iterdir()}

    completed, config = replay_existing_painting(
        source,
        output,
        output_size=128,
        supersample=2,
    )

    assert completed == output
    assert {path.name for path in output.iterdir()} == {
        "reference.png",
        "initial.png",
        "best.png",
        "final.png",
        "painting.gif",
        "replay_config.json",
    }
    for filename in ("reference.png", "initial.png", "best.png", "final.png"):
        with Image.open(output / filename) as image:
            assert image.size == (128, 128)
    saved = json.loads((output / "replay_config.json").read_text(encoding="utf-8"))
    assert saved == config
    assert config["source_best_step"] == 1
    assert config["source_best_step_source"] == "summary.json"
    assert config["stroke_sequence_changed"] is False
    assert config["models_loaded"] is False
    assert config["models_trained"] is False
    assert config["source_artifacts_unchanged"] is True
    assert source_before == {path.name: path.read_bytes() for path in source.iterdir()}


def test_legacy_summary_derives_best_step_from_progress(tmp_path: Path) -> None:
    source = _source_painting(tmp_path)
    (source / "summary.json").write_text(
        json.dumps({"method": "learned"}),
        encoding="utf-8",
    )

    _, config = replay_existing_painting(
        source,
        tmp_path / "legacy-128",
        output_size=128,
        supersample=2,
    )

    assert config["source_best_step"] == 1
    assert config["source_best_step_source"] == "progress.csv"


def test_replay_refuses_to_overwrite_output(tmp_path: Path) -> None:
    source = _source_painting(tmp_path)
    output = tmp_path / "painting-128"
    output.mkdir()

    with pytest.raises(FileExistsError, match="overwrite"):
        replay_existing_painting(
            source,
            output,
            output_size=128,
        )
