import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageDraw

from latent_stroke_dynamics.painting_cli import (
    normalize_target_polarity,
    paint_target,
)
from latent_stroke_dynamics.renderer import Stroke, blank_canvas, render_stroke


def save_test_target(path: Path) -> None:
    target = render_stroke(
        blank_canvas(32),
        Stroke(0.1, 0.2, 0.9, 0.8, width=3, value=32),
    )
    target.save(path)


@pytest.mark.parametrize(
    ("background", "foreground", "expected_inversion"),
    [(0, 255, True), (255, 0, False)],
)
def test_auto_polarity_maps_foreground_to_dark_on_white(
    background: int,
    foreground: int,
    expected_inversion: bool,
) -> None:
    image = Image.new("L", (64, 64), color=background)
    draw = ImageDraw.Draw(image)
    draw.ellipse((20, 12, 44, 52), fill=foreground)

    normalized, inverted, border_median = normalize_target_polarity(
        image,
        polarity="auto",
    )

    values = np.asarray(normalized)
    assert inverted is expected_inversion
    assert border_median == float(background)
    assert values[0, 0] == 255
    assert values[32, 32] == 0


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
        "processed_target_before_polarity.png",
        "processed_target.png",
        "initial_canvas.png",
        "best_painting.png",
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
    assert config["target_processing"]["polarity_inverted"] is False
    assert summary["method"] == "exact"
    assert summary["strokes"] == 2
    assert summary["target_polarity_inverted"] is False
    assert 0 <= summary["best_step"] <= 2
    assert summary["best_mse"] <= summary["final_mse"] + 1e-12
    assert summary["final_mse_ratio_to_best"] >= 1.0 - 1e-12


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
