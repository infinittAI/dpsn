from __future__ import annotations

import numpy as np
import openslide

from ai.wsi.patch import Patch
from ai.wsi.patch_ref import PatchRef


def load_patch(ref: PatchRef) -> Patch:
    """
    Load a single patch from a WSI using the metadata stored in PatchRef.

    Returns
    -------
    Patch
        Patch image data in [C, H, W] RGB uint8 format.
    """
    # OpenSlide expects:
    # - location in level-0 coordinates
    # - size in pixels at ref.read_level
    with openslide.OpenSlide(str(ref.image_path)) as slide: # opens the WSI file and reads one patch from it
        region = slide.read_region(
            ref.level0_pos, # Top-left patch location in level-0 coordinates
            ref.read_level, # Which level to read from
            ref.read_size,  # Patch size in pixels at read_level
        )

    # OpenSlide returns a PIL image in RGBA. Convert to RGB - drop alpha channel
    rgb = region.convert("RGB")

    # PIL gives HWC (height, width, channel); Patch expects [C, H, W], so transpose the axes.
    img_hwc = np.asarray(rgb, dtype=np.uint8)
    img_chw = np.transpose(img_hwc, (2, 0, 1))

    return Patch(
        ref=ref,
        img=img_chw,
    )
