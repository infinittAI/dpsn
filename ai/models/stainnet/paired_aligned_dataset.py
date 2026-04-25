from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class PairedAlignedImageDataset(Dataset):
    """
    Paired aligned dataset for StainNet training.

    Each source image must have a matching target image with the same filename.
    The dataset returns RGB tensors in CHW format normalized to [-1, 1], which
    matches the original StainNet training/inference convention.
    """

    def __init__(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        image_size: int = 256,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.image_size = int(image_size)

        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        if not self.target_dir.is_dir():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")

        source_files = self._list_images(self.source_dir)
        target_files = self._list_images(self.target_dir)

        source_map = {path.name: path for path in source_files}
        target_map = {path.name: path for path in target_files}

        missing_in_target = sorted(set(source_map) - set(target_map))
        missing_in_source = sorted(set(target_map) - set(source_map))
        if missing_in_target or missing_in_source:
            details: list[str] = []
            if missing_in_target:
                details.append(f"missing target files: {missing_in_target[:10]}")
            if missing_in_source:
                details.append(f"missing source files: {missing_in_source[:10]}")
            raise ValueError(
                "Source/target folders must contain the same filenames; "
                + "; ".join(details)
            )

        self.filenames = sorted(source_map.keys())
        self.source_map = source_map
        self.target_map = target_map

        if not self.filenames:
            raise ValueError(
                f"No supported images found in {self.source_dir} and {self.target_dir}"
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, str]: # get the source and target file of corresponding index
        filename = self.filenames[index]
        source = self._load_image(self.source_map[filename])
        target = self._load_image(self.target_map[filename])
        return source, target, filename
    
    # looks inside a folder and returns a sorted list of image file paths
    def _list_images(self, directory: Path) -> list[Path]:
        return sorted(
            [
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS #.lower() treats PNG, png Png as the same
            ]
        )
    
    # loads one image file from disk and converts it into the exact numeric format the model wants
    def _load_image(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB") # open and force into RGB format
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR) #resizes the image to a fixed square size(256 x 256)
        image_np = np.asarray(image, dtype=np.float32) / 255.0 # converts the PIL image into a NumPy array (dividing by 255 converts them to values 0-1)
        image_np = np.transpose(image_np, (2, 0, 1)) # Change from HWC to CHW
        image_np = (image_np - 0.5) * 2.0 # Normalize from [0,1] to [-1,1], because that's what stainnet expects
        return image_np.astype(np.float32) # return processed image
