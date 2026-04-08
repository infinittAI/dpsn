from dataclasses import dataclass

import numpy as np

from ai.wsi.patch_ref import PatchRef

@dataclass
class Patch:
    ref: PatchRef
    img: np.ndarray # [C, H, W]

    def __post_init__(self) -> None:
      if not isinstance(self.ref, PatchRef): #should be a PatchRef
          raise TypeError(f"ref must be a PatchRef, got {type(self.ref).__name__}")

      if not isinstance(self.img, np.ndarray): # should be an array
          raise TypeError(f"img must be a numpy.ndarray, got {type(self.img).__name__}")

      if self.img.ndim != 3: #should have 3 dimensions
          raise ValueError(f"img must have 3 dimensions [C, H, W], got shape {self.img.shape}")

      if self.img.shape[0] != 3:
          raise ValueError(
              f"img must have 3 channels in CHW format, got shape {self.img.shape}"
          )

      if self.img.shape[1] <= 0 or self.img.shape[2] <= 0:
          raise ValueError(f"img height and width must be > 0, got shape {self.img.shape}")
