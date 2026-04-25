from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Job: to provide unpaired samples from two image domains for StainGAN training.
class UnpairedDomainImageDataset(Dataset):
    """
    Unpaired two-domain dataset for CycleGAN-style StainGAN training.

    Domain A and Domain B are sampled independently. This matches the StainGAN
    setup referenced by the StainNet paper, where StainGAN is trained as an
    unpaired image translation model between scanner domains.
    """

    def __init__(
        self,
        domain_a_dir: str | Path,
        domain_b_dir: str | Path,
        image_size: int = 256,
        recursive: bool = True, # whether to search subfolders recursively
    ) -> None:
        self.domain_a_dir = Path(domain_a_dir)
        self.domain_b_dir = Path(domain_b_dir)
        self.image_size = int(image_size)
        self.recursive = bool(recursive)

        if not self.domain_a_dir.is_dir():
            raise FileNotFoundError(f"Domain A directory not found: {self.domain_a_dir}")
        if not self.domain_b_dir.is_dir():
            raise FileNotFoundError(f"Domain B directory not found: {self.domain_b_dir}")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be > 0, got {self.image_size}")

        self.domain_a_files = self._list_images(self.domain_a_dir)
        self.domain_b_files = self._list_images(self.domain_b_dir)

        if not self.domain_a_files:
            raise ValueError(f"No supported images found in {self.domain_a_dir}")
        if not self.domain_b_files:
            raise ValueError(f"No supported images found in {self.domain_b_dir}")

    def __len__(self) -> int:
        return max(len(self.domain_a_files), len(self.domain_b_files))
    
    # returns one training sample
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, str, str]:
        path_a = self.domain_a_files[index % len(self.domain_a_files)] # choose Domain A image deterministically
        path_b = random.choice(self.domain_b_files) # Choose Domain B image randomly
        image_a = self._load_image(path_a) # load both images: reads the files and converts them into normalized NumPy arrays
        image_b = self._load_image(path_b)
        return image_a, image_b, path_a.name, path_b.name
    
    # Scans a directory and returns all valid image files
    def _list_images(self, directory: Path) -> list[Path]:
        pattern = "**/*" if self.recursive else "*"
        return sorted(
            [
                path
                for path in directory.glob(pattern)
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ]
        )
    
    # Loads one image and converts it into the format expected by the model
    def _load_image(self, path: Path) -> np.ndarray:
        image = Image.open(path).convert("RGB") # ensure 3-channel RGB input
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR) # resize to 256x256
        image_np = np.asarray(image, dtype=np.float32) / 255.0 # Convert to NumPy and scale to [0,1]
        image_np = np.transpose(image_np, (2, 0, 1))
        image_np = (image_np - 0.5) * 2.0
        return image_np.astype(np.float32)
