from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from pathlib import PosixPath, WindowsPath
import time
from typing import Any

import numpy as np
import torch

from ai.metrics.ssim import SSIM
from ai.models.stainnet.stainnet_model import StainNet as StainNetModel
from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.samplers.grid_sampler import GridSampler
from ai.wsi.handle import WSIHandle, open_wsi_handle
from ai.wsi.loader import load_patch
from ai.wsi.writer import TiffWSIWriter

# class that stores all settings needed for StainNet WSI inference
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
    tile_size: int = 512 # small rectangular chunks called that a WSI is divided and saved in
    pyramid_levels: int = 2
    device: str = "auto"
    compression: str | None = None # whether to compress the output TIFF, and by which method
    keep_store: bool = False # whether to keep the writer’s intermediate storage after output is finalized
    verbose: bool = False
    log_every_batches: int = 10
    compute_ssim: bool = True

# Class that performs the whole inference procedure on a WSI
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
        total_refs = len(refs)
        total_batches = (total_refs + self.config.batch_size - 1) // self.config.batch_size

        checkpoint_path = self._resolve_checkpoint_path()
        self._log_run_summary(
            src_img_path=src_img_path,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            wsi_handle=src_wsi_handle,
            total_refs=total_refs,
            total_batches=total_batches,
        )

        self._log(
            f"Loaded WSI metadata: read_level={self.config.read_level}, "
            f"level_shape=({read_w}, {read_h}), total_patches={total_refs}, "
            f"batch_size={self.config.batch_size}, total_batches={total_batches}"
        )

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

        run_start = time.time()
        for start in range(0, len(refs), self.config.batch_size):
            batch_refs = refs[start:start + self.config.batch_size]
            batch_patches = [load_patch(ref).img for ref in batch_refs]
            normalized_batch = self._normalize_batch(batch_patches)

            for ref, normalized_patch in zip(batch_refs, normalized_batch):
                writer.write_patch(ref, normalized_patch)

            batch_index = (start // self.config.batch_size) + 1
            if (
                batch_index == 1
                or batch_index == total_batches
                or batch_index % max(self.config.log_every_batches, 1) == 0
            ):
                elapsed = time.time() - run_start
                processed = min(start + len(batch_refs), total_refs)
                rate = processed / elapsed if elapsed > 0 else 0.0
                remaining = total_refs - processed
                eta_seconds = remaining / rate if rate > 0 else float("inf")
                eta_text = (
                    f"{eta_seconds:.1f}s"
                    if np.isfinite(eta_seconds)
                    else "unknown"
                )
                self._log(
                    f"Processed batch {batch_index}/{total_batches} "
                    f"({processed}/{total_refs} patches, "
                    f"{rate:.2f} patches/s, eta {eta_text})"
                )

        self._log("Finalizing TIFF writer...")
        final_output_path = writer.finalize()
        total_elapsed = time.time() - run_start
        ssim_score = None
        if self.config.compute_ssim:
            self._log("Computing SSIM against the input WSI...")
            normalized_wsi_handle = open_wsi_handle(final_output_path)
            ssim_score = self._compute_ssim(
                origin_image=src_wsi_handle,
                normalized_image=normalized_wsi_handle,
            )

        self._log(
            f"Finished inference in {total_elapsed:.1f}s. "
            f"Output written to {final_output_path}"
        )
        if ssim_score is not None:
            self._log(f"SSIM: {ssim_score:.6f}")
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
        checkpoint = self._load_checkpoint(checkpoint_path)
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_module_prefix(state_dict)
        model.load_state_dict(state_dict)
        return model

    def _load_checkpoint(self, checkpoint_path: Path) -> Any:
        try:
            return torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except pickle.UnpicklingError as exc:
            # Our own training checkpoints store config metadata containing
            # pathlib paths, which strict weights-only loading blocks unless
            # those classes are explicitly allowlisted.
            torch.serialization.add_safe_globals([Path, PosixPath, WindowsPath])
            try:
                return torch.load(
                    checkpoint_path,
                    map_location=self.device,
                    weights_only=True,
                )
            except pickle.UnpicklingError:
                raise ValueError(
                    "Failed to load the StainNet checkpoint in weights-only mode. "
                    "If this checkpoint was produced outside this project, inspect "
                    "its contents before relaxing the loader further."
                ) from exc

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

    def _compute_ssim(
        self,
        origin_image: WSIHandle,
        normalized_image: WSIHandle,
    ) -> float:
        metric = SSIM(
            patch_size=self.config.patch_size,
            stride=self.config.stride,
            read_level=self.config.read_level,
        )
        return metric.evaluate(origin_image, normalized_image)

    def _log_run_summary(
        self,
        src_img_path: Path,
        checkpoint_path: Path,
        output_path: Path,
        wsi_handle: WSIHandle,
        total_refs: int,
        total_batches: int,
    ) -> None:
        self._log("Run configuration:")
        self._log(f"  input_path={src_img_path}")
        self._log(f"  checkpoint_path={checkpoint_path}")
        self._log(f"  output_path={output_path}")
        self._log(f"  read_level={self.config.read_level}")
        self._log(f"  patch_size={self.config.patch_size}")
        self._log(f"  stride={self.config.stride}")
        self._log(f"  batch_size={self.config.batch_size}")
        self._log(f"  tile_size={self.config.tile_size}")
        self._log(f"  pyramid_levels={self.config.pyramid_levels}")
        self._log(f"  compression={self.config.compression}")
        self._log(f"  device={self.device}")
        self._log(f"  compute_ssim={self.config.compute_ssim}")
        self._log(f"  level_dimensions={wsi_handle.level_dimensions}")
        self._log(f"  total_patches={total_refs}")
        self._log(f"  total_batches={total_batches}")

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[StainNetPipeline] {message}", flush=True)


# Backward-compatible alias while the rest of the project catches up.
StainNet = StainNetPipeline
StainNetConfig = StainNetInferenceConfig
