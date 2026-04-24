import numpy as np
from pathlib import Path
from PIL import Image
import zarr

from ai.wsi.patch_ref import PatchRef

class ZarrWSIWriter:
    def __init__(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        channels: int = 3,
        tile_size: int = 512,
        overwrite: bool = True
    ) -> None:
        self.output_path = Path(output_path)
        self.width = width
        self.height = height
        self.channels = channels
        self.tile_size = tile_size
        
        mode = "w" if overwrite else "a"

        self.root = zarr.open_group(str(self.output_path), mode=mode)

        self.image = self.root.create_array(
            name="image",
            shape=(self.height, self.width, self.channels),
            chunks=(self.tile_size, self.tile_size, self.channels),
            dtype=np.uint8,
            fill_value=0
        )

        self.metadata = {
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "tile_size": self.tile_size,
            "dtype": str(np.uint8)
        }

        self.root.attrs.update(self.metadata)


    def write_patch(self, ref: PatchRef, img: np.ndarray) -> None:
        x1 = int(ref.x / ref.downsample)
        y1 = int(ref.y / ref.downsample)

        if x1 >= self.width or y1 >= self.height:
            return
        
        if x1 < 0 or y1 < 0:
            raise ValueError()
        
        patch_h, patch_w = img.shape[1:]

        x2 = min(x1 + patch_w, self.width)
        y2 = min(y1 + patch_h, self.height)

        write_w = x2 - x1
        write_h = y2 - y1

        if write_w <= 0 or write_h <= 0:
            return
        
        cropped = (img[:, :write_h, :write_w]).astype(np.uint8)
        self.image[y1:y2, x1:x2, :] = cropped.transpose([1, 2, 0])
    
    def get_thumbnail(self, max_size: int = 1024):
        arr = self.root["image"]
        H, W, C = arr.shape

        stride = max(1, int(max(H / max_size, W / max_size)))

        thumb = arr[::stride, ::stride, :]

        return Image.fromarray(thumb.astype("uint8"), mode="RGB")
