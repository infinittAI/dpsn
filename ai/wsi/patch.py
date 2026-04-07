from dataclasses import dataclass

import numpy as np

from ai.wsi.patch_ref import PatchRef

@dataclass
class Patch:
    ref: PatchRef
    img: np.ndarray # [C, H, W]
