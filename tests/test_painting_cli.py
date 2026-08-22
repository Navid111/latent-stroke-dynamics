import json
from pathlib import Path

import pandas as pd
import pytest

from latent_stroke_dynamics.painting_cli import paint_target
from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke


def save_test_target(path: Path) -> None:
    target = render_stroke(
        blank_canvas(32),
        Stroke(0.1, 0.2, 0.9, 0.8, width=3, value=32),
    )
    target.save(path)


def test_exact_qualitative_command_saves_complete_artifact_set(tmp_path) -> None:
    target_path = tmp_path / "target.png"
    output_dir = tmp_path / "painting"
    save_test_target(target_path)

    completed_dir, summary = paint_target(
        target_path=target_path,
        output_dir=output_dir,
        method="exact",
        strokes=2,
        candidates=8,
        seed=91,
        gif_scale=1,
    )

    assert completed_dir == output_dir
    assert output_dir.is_dir()
    required = {
        "processed_target.png",
        "initial_canvas.png",
        "final_painting.png",
        "progress.csv",
        "summary.csv",
        "summary.json",
        "strokes.json",
        "run_config.json",
        "painting.gif",
        "progress.png",
        "comparison.png",
    }
    assert required.issubset({path.name for path in output_dir.iterdir()})
    assert len(list((output_dir / "frames").glob("frame_*.png"))) == 3

    progress = pd.read_csv(output_dir / "progress.csv")
    assert len(progress) == 2
    assert list(progress["step"]) == [1, 2]
    strokes = json.loads((output_dir / "strokes.json").read_text())
    assert len(strokes) == 2
    config = json.loads((output_dir / "run_config.json").read_text())
    assert config["qualitative_demo"] is True
    assert config["controlled_stage3_result_unchanged"] is True
    assert config["retraining_performed"] is False
    assert config["checkpoint_path"] is None
    assert summary["method"] == "exact"
    assert summary["strokes"] == 2


def test_painter_refuses_to_overwrite_complete_output(tmp_path) -> None:
    target_path = tmp_path / "target.png"
    output_dir = tmp_path / "painting"
    save_test_target(target_path)
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        paint_target(
            target_path=target_path,
            output_dir=output_dir,
            method="exact",
            strokes=1,
            candidates=4,
        )


def test_painter_preserves_existing_incomplete_output(tmp_path) -> None:
    target_path = tmp_path / "target.png"
    output_dir = tmp_path / "painting"
    save_test_target(target_path)
    incomplete = tmp_path / "painting.incomplete"
    incomplete.mkdir()
    marker = incomplete / "failure.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="incomplete"):
        paint_target(
            target_path=target_path,
            output_dir=output_dir,
            method="exact",
            strokes=1,
            candidates=4,
        )
    assert marker.read_text(encoding="utf-8") == "preserve me"
