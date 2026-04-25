from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dpsn.ai.models.stainnet.stainnet_model import StainNet as StainNetModel
from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.samplers.grid_sampler import GridSampler
from ai.wsi.handle import open_wsi_handle
from ai.wsi.loader import load_patch
from ai.wsi.writer import TiffWSIWriter


@dataclass(slots=True)
class StainNetInferenceConfig:
    """
    Configuration for patch-wise WSI inference with a trained StainNet model.
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
    tile_size: int = 512
    pyramid_levels: int = 2
    device: str = "auto"
    compression: str | None = None
    keep_store: bool = False


class StainNetPipeline(ModelPipeline):
    """
    WSI inference pipeline for a trained StainNet model.

    This path is for inference only. Training uses paired aligned image/patch
    folders and lives in separate dataset/training modules.
    """

    def __init__(self, config: StainNetInferenceConfig | None = None) -> None:
        super().__init__()
        self.config = config or StainNetInferenceConfig()
        self._validate_config()

        self.device = self._select_device(self.config.device)
        self.grid_sampler = GridSampler(
            patch_size=self.config.patch_size,
            stride=self.config.stride,
            read_level=self.config.read_level,
        )
        self.model = self._load_model().to(self.device)
        self.model.eval()

    def run(
        self,
        src_img_path: Path,
        target_img_path: Path | None = None,
    ) -> PipelineResult:
        del target_img_path

        src_wsi_handle = open_wsi_handle(src_img_path)
        level_count = len(src_wsi_handle.level_dimensions)
        if not (0 <= self.config.read_level < level_count):
            raise ValueError(
                f"read_level {self.config.read_level} must be within [0, {level_count - 1}]"
            )

        read_w, read_h = src_wsi_handle.level_dimensions[self.config.read_level]
        level_downsample = float(
            src_wsi_handle.level_downsamples[self.config.read_level]
        )
        refs = self.grid_sampler.sample(src_wsi_handle)
        output_path = self._build_output_path(src_img_path)

        writer = TiffWSIWriter(
            output_path=output_path,
            width=read_w,
            height=read_h,
            level_downsample=level_downsample,
            channels=3,
            tile_size=self.config.tile_size,
            overwrite=True,
            pyramid_levels=self.config.pyramid_levels,
            compression=self.config.compression,
            mpp_x=src_wsi_handle.mpp[0] if src_wsi_handle.mpp[0] > 0 else None,
            mpp_y=src_wsi_handle.mpp[1] if src_wsi_handle.mpp[1] > 0 else None,
            keep_store=self.config.keep_store,
        )

        for start in range(0, len(refs), self.config.batch_size):
            batch_refs = refs[start:start + self.config.batch_size]
            batch_patches = [load_patch(ref).img for ref in batch_refs]
            normalized_batch = self._normalize_batch(batch_patches)

            for ref, normalized_patch in zip(batch_refs, normalized_batch):
                writer.write_patch(ref, normalized_patch)

        final_output_path = writer.finalize()
        return PipelineResult(output_path=str(final_output_path))

    def _validate_config(self) -> None:
        if self.config.patch_size <= 0:
            raise ValueError("patch_size must be > 0")
        if self.config.stride <= 0:
            raise ValueError("stride must be > 0")
        if self.config.read_level < 0:
            raise ValueError("read_level must be >= 0")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.config.tile_size <= 0:
            raise ValueError("tile_size must be > 0")
        if self.config.pyramid_levels < 0:
            raise ValueError("pyramid_levels must be >= 0")

    def _load_model(self) -> StainNetModel:
        model = StainNetModel(
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc,
            n_layer=self.config.n_layer,
            n_channel=self.config.channels,
            kernel_size=self.config.kernel_size,
        )

        checkpoint_path = self._resolve_checkpoint_path()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
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


# Backward-compatible alias while the rest of the project catches up.
StainNet = StainNetPipeline
StainNetConfig = StainNetInferenceConfig
