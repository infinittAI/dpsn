from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import zarr

from ai.wsi.patch_ref import PatchRef


class RegularTiffWSIWriter:
    """
    Assemble read-level patches into a single regular TIFF image.

    PatchRef stores x/y in level-0 coordinates, so this writer needs the
    read-level downsample to map each patch back onto the output canvas.
    """

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        level_downsample: float,
        channels: int = 3,
        overwrite: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {(width, height)}")
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if level_downsample <= 0:
            raise ValueError(
                f"level_downsample must be > 0, got {level_downsample}"
            )

        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.channels = channels
        self.level_downsample = float(level_downsample)

        if self.output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {self.output_path}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.image = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)

    def write_patch(self, ref: PatchRef, img: np.ndarray) -> None:
        x1, y1 = self._read_level_pos(ref)

        if x1 >= self.width or y1 >= self.height:
            return
        if x1 < 0 or y1 < 0:
            raise ValueError(f"Negative write position: {(x1, y1)}")

        img_hwc = self._to_hwc_uint8(img)
        patch_h, patch_w = img_hwc.shape[:2]

        x2 = min(x1 + patch_w, self.width)
        y2 = min(y1 + patch_h, self.height)
        write_w = x2 - x1
        write_h = y2 - y1

        if write_w <= 0 or write_h <= 0:
            return

        self.image[y1:y2, x1:x2, :] = img_hwc[:write_h, :write_w, :]

    def finalize(self) -> Path:
        Image.fromarray(self.image, mode="RGB").save(self.output_path, format="TIFF")
        return self.output_path

    def _read_level_pos(self, ref: PatchRef) -> tuple[int, int]:
        x = int(round(ref.x / self.level_downsample))
        y = int(round(ref.y / self.level_downsample))
        return (x, y)

    def _to_hwc_uint8(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError(f"img must be a numpy.ndarray, got {type(img).__name__}")
        if img.ndim != 3:
            raise ValueError(f"img must have shape [C, H, W], got {img.shape}")
        if img.shape[0] != self.channels:
            raise ValueError(
                f"img must have {self.channels} channels in CHW format, got {img.shape}"
            )
        if img.dtype != np.uint8:
            raise ValueError(f"img must be uint8, got {img.dtype}")

        return np.transpose(img, (1, 2, 0))


class ZarrWSIWriter:
    """
    Optional chunked writer for larger slide outputs.

    Kept for future use, but updated to share the same coordinate handling as
    the TIFF writer so PatchRef works correctly here too.
    """

    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        level_downsample: float,
        channels: int = 3,
        tile_size: int = 512,
        overwrite: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"width and height must be > 0, got {(width, height)}")
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if tile_size <= 0:
            raise ValueError(f"tile_size must be > 0, got {tile_size}")
        if level_downsample <= 0:
            raise ValueError(
                f"level_downsample must be > 0, got {level_downsample}"
            )

        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.channels = channels
        self.tile_size = tile_size
        self.level_downsample = float(level_downsample)

        mode = "w" if overwrite else "a"
        self.root = zarr.open_group(str(self.output_path), mode=mode)
        self.image = self.root.create_array(
            name="image",
            shape=(self.height, self.width, self.channels),
            chunks=(self.tile_size, self.tile_size, self.channels),
            dtype=np.uint8,
            fill_value=0,
        )

        self.metadata = {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "tile_size": self.tile_size,
            "dtype": str(np.uint8),
            "level_downsample": self.level_downsample,
        }
        self.root.attrs.update(self.metadata)

    def write_patch(self, ref: PatchRef, img: np.ndarray) -> None:
        x1, y1 = self._read_level_pos(ref)

        if x1 >= self.width or y1 >= self.height:
            return
        if x1 < 0 or y1 < 0:
            raise ValueError(f"Negative write position: {(x1, y1)}")

        img_hwc = self._to_hwc_uint8(img)
        patch_h, patch_w = img_hwc.shape[:2]

        x2 = min(x1 + patch_w, self.width)
        y2 = min(y1 + patch_h, self.height)
        write_w = x2 - x1
        write_h = y2 - y1

        if write_w <= 0 or write_h <= 0:
            return

        self.image[y1:y2, x1:x2, :] = img_hwc[:write_h, :write_w, :]

    def finalize(self) -> Path:
        return self.output_path

    def _read_level_pos(self, ref: PatchRef) -> tuple[int, int]:
        x = int(round(ref.x / self.level_downsample))
        y = int(round(ref.y / self.level_downsample))
        return (x, y)

    def _to_hwc_uint8(self, img: np.ndarray) -> np.ndarray:
        if not isinstance(img, np.ndarray):
            raise TypeError(f"img must be a numpy.ndarray, got {type(img).__name__}")
        if img.ndim != 3:
            raise ValueError(f"img must have shape [C, H, W], got {img.shape}")
        if img.shape[0] != self.channels:
            raise ValueError(
                f"img must have {self.channels} channels in CHW format, got {img.shape}"
            )
        if img.dtype != np.uint8:
            raise ValueError(f"img must be uint8, got {img.dtype}")

        return np.transpose(img, (1, 2, 0))
