from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from ai.models.stainnet_model import StainNet
from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.wsi.handle import WSIHandle, open_wsi_handle
from ai.wsi.loader import load_patch
from ai.wsi.patch_ref import PatchRef


@dataclass(slots=True)
class StainNetConfig:
    """
    Runtime configuration for StainNet WSI inference.

    Defaults follow the requested first implementation:
    - full-grid inference
    - regular TIFF output
    - patch_size=512, stride=512, read_level=0, batch_size=8
    """

    checkpoint_path: Path | None = None
    checkpoint_dir: Path = Path(__file__).resolve().parents[1] / "checkpoints"
    output_dir: Path = Path("result_stainnet")

    input_nc: int = 3
    output_nc: int = 3
    channels: int = 32
    n_layer: int = 3
    kernel_size: int = 1

    patch_size: int = 512
    stride: int = 512
    read_level: int = 0
    batch_size: int = 8
    device: str = "auto"


class StainNetPipeline(ModelPipeline):
    """
    StainNet stain-normalization pipeline for WSI files.

    This pipeline follows the original StainNet inference convention:
    - input tensor is RGB in [0, 1]
    - normalize to [-1, 1] before the model
    - convert model output back to [0, 1]
    - clamp and save as uint8 RGB

    StainNet is a learned fixed-domain mapping, so target_img_path is ignored.
    The checkpoint itself defines the target stain style.
    """

    def __init__(self, config: StainNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or StainNetConfig()
        self._validate_config()

        self.device = self._select_device(self.config.device)
        self.model = self._load_model().to(self.device)
        self.model.eval()

    def run(
        self,
        src_img_path: Path,
        target_img_path: Path | None = None,
    ) -> PipelineResult:
        del target_img_path  # StainNet inference uses a trained target domain.

        src_wsi_handle = open_wsi_handle(src_img_path)
        refs_with_pos = self._make_full_grid_refs(src_wsi_handle)

        read_w, read_h = src_wsi_handle.level_dimensions[self.config.read_level]
        output_hwc = np.zeros((read_h, read_w, 3), dtype=np.uint8)

        for start in range(0, len(refs_with_pos), self.config.batch_size):
            batch_items = refs_with_pos[start:start + self.config.batch_size]
            batch_refs = [item[0] for item in batch_items]
            batch_patches = [load_patch(ref).img for ref in batch_refs]

            normalized_batch = self._normalize_batch(batch_patches)

            for normalized_chw, (_, x, y) in zip(normalized_batch, batch_items):
                patch_h = normalized_chw.shape[1]
                patch_w = normalized_chw.shape[2]
                output_hwc[y:y + patch_h, x:x + patch_w, :] = np.transpose(
                    normalized_chw,
                    (1, 2, 0),
                )

        output_path = self._build_output_path(src_img_path)
        Image.fromarray(output_hwc, mode="RGB").save(output_path, format="TIFF")

        return PipelineResult(output_path=str(output_path))

    def _validate_config(self) -> None:
        if self.config.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if self.config.stride <= 0:
            raise ValueError("stride must be > 0")
        if self.config.read_level < 0:
            raise ValueError("read_level must be >= 0")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")

    def _load_model(self) -> StainNet:
        model = StainNet(
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc,
            n_layer=self.config.n_layer,
            n_channel=self.config.channels,
            kernel_size=self.config.kernel_size,
        )

        checkpoint_path = self._resolve_checkpoint_path()
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_module_prefix(state_dict)
        model.load_state_dict(state_dict)
        return model

    def _resolve_checkpoint_path(self) -> Path:
        if self.config.checkpoint_path is not None:
            checkpoint_path = Path(self.config.checkpoint_path)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"StainNet checkpoint not found: {checkpoint_path}")
            return checkpoint_path

        checkpoint_dir = Path(self.config.checkpoint_dir)
        candidates = sorted(
            [
                *checkpoint_dir.glob("*.pth"),
                *checkpoint_dir.glob("*.pt"),
            ]
        )

        if not candidates:
            raise FileNotFoundError(
                f"No StainNet checkpoint found in: {checkpoint_dir}"
            )
        if len(candidates) > 1:
            names = ", ".join(str(path) for path in candidates)
            raise ValueError(
                "Multiple checkpoint files found. Pass checkpoint_path explicitly: "
                f"{names}"
            )

        return candidates[0]

    def _extract_state_dict(self, checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "net", "model"):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value

            if all(isinstance(key, str) for key in checkpoint.keys()):
                return checkpoint

        raise ValueError("Checkpoint does not contain a valid state_dict.")

    def _strip_module_prefix(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        prefix = "module."
        if not any(key.startswith(prefix) for key in state_dict):
            return state_dict

        return {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        }

    def _make_full_grid_refs(
        self,
        wsi_handle: WSIHandle,
    ) -> list[tuple[PatchRef, int, int]]:
        level_count = len(wsi_handle.level_dimensions)
        if not (0 <= self.config.read_level < level_count):
            raise ValueError(
                f"read_level {self.config.read_level} must be within [0, {level_count - 1}]"
            )

        read_w, read_h = wsi_handle.level_dimensions[self.config.read_level]
        patch_w = min(self.config.patch_size, read_w)
        patch_h = min(self.config.patch_size, read_h)

        xs = self._grid_positions(read_w, patch_w, self.config.stride)
        ys = self._grid_positions(read_h, patch_h, self.config.stride)

        refs: list[tuple[PatchRef, int, int]] = []
        for y in ys:
            for x in xs:
                ref = wsi_handle.make_ref(
                    pos=(x, y),
                    level=self.config.read_level,
                    dim=(patch_w, patch_h),
                )
                refs.append((ref, x, y))

        return refs

    def _grid_positions(
        self,
        length: int,
        patch_length: int,
        stride: int,
    ) -> list[int]:
        max_start = length - patch_length
        positions = list(range(0, max_start + 1, stride))

        # Include the right/bottom edge even when stride does not land exactly
        # on the final valid top-left coordinate.
        if not positions or positions[-1] != max_start:
            positions.append(max_start)

        return positions

    def _normalize_batch(self, patches_chw: list[np.ndarray]) -> list[np.ndarray]:
        batch = np.stack(patches_chw, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(batch).to(self.device)

        # Original StainNet test code maps [0, 1] -> [-1, 1] before inference.
        tensor = (tensor - 0.5) * 2.0

        with torch.inference_mode():
            output = self.model(tensor)

        # Original StainNet test code maps model output back to [0, 1].
        output = output * 0.5 + 0.5
        output = torch.clamp(output, 0.0, 1.0)

        output_np = output.detach().cpu().numpy()
        output_np = np.rint(output_np * 255.0).astype(np.uint8)
        return [output_np[i] for i in range(output_np.shape[0])]

    def _select_device(self, device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_output_path(self, src_img_path: Path) -> Path:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(src_img_path).stem
        return self.config.output_dir / f"{stem}_stainnet.tiff"
