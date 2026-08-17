"""Frozen visual-encoder wrapper exposing global and spatial features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel


@dataclass(frozen=True)
class EncodingBatch:
    """Normalized global and patch-token representations on CPU."""

    global_features: torch.Tensor
    patch_features: torch.Tensor
    patch_grid: tuple[int, int]


class FrozenVisionEncoder:
    """Load a pretrained Hugging Face vision model and never update its weights."""

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        device: str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = model_name

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(
        self,
        images: Sequence[Image.Image],
        batch_size: int = 16,
    ) -> EncodingBatch:
        if not images:
            raise ValueError("At least one image is required.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        global_batches: list[torch.Tensor] = []
        patch_batches: list[torch.Tensor] = []
        patch_grid: tuple[int, int] | None = None

        for start in tqdm(range(0, len(images), batch_size), desc="Encoding canvases"):
            batch = [image.convert("RGB") for image in images[start : start + batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {name: value.to(self.device) for name, value in inputs.items()}

            outputs = self.model(**inputs)
            hidden = outputs.last_hidden_state

            patch_size = int(getattr(self.model.config, "patch_size", 14))
            input_height, input_width = inputs["pixel_values"].shape[-2:]
            rows = input_height // patch_size
            columns = input_width // patch_size
            expected_patch_count = rows * columns
            if hidden.shape[1] < expected_patch_count + 1:
                raise ValueError(
                    "The selected model does not expose the expected class and patch tokens."
                )

            # Taking the final expected tokens tolerates models that insert register tokens
            # between the class token and spatial patch tokens.
            global_features = F.normalize(hidden[:, 0, :], dim=-1)
            patch_features = F.normalize(hidden[:, -expected_patch_count:, :], dim=-1)

            global_batches.append(global_features.cpu())
            patch_batches.append(patch_features.cpu())
            patch_grid = (rows, columns)

        assert patch_grid is not None
        return EncodingBatch(
            global_features=torch.cat(global_batches, dim=0),
            patch_features=torch.cat(patch_batches, dim=0),
            patch_grid=patch_grid,
        )
