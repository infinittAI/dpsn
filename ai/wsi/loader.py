from __future__ import annotations
from pathlib import Path

import numpy as np
import openslide

from ai.wsi.handle import WSIHandle
from ai.wsi.loaders.openslide_loader import OpenSlideLoader
from ai.wsi.loaders.tifffile_loader import TiffFileLoader
from ai.wsi.patch import Patch
from ai.wsi.patch_ref import PatchRef


# 사용자로부터 Image path를 전달받아서, 
# 다양한 wsi format(.tif ...)에 따라 WSIHandle을 구성하고 리턴
def open_wsi_handle(image_path: str | Path) -> WSIHandle:
    image_path = Path(image_path)
    
    try:
        return OpenSlideLoader.open_wsi_handle(image_path)
    except:
        try:
            return TiffFileLoader.open_wsi_handle(image_path)
        except:
            raise ValueError()


def load_patch(ref: PatchRef) -> Patch:
    """
    Load a single patch from a WSI using the metadata stored in PatchRef.

    Returns
    -------
    Patch
        Patch image data in [C, H, W] RGB uint8 format.
    """
    try:
        return OpenSlideLoader.load_patch(ref)
    except:
        try:
            return TiffFileLoader.load_patch(ref)
        except:
            raise ValueError()


def load_patches_from_image(refs: list[PatchRef], image_path: str | Path) -> list[Patch]:
    patches = []
    with openslide.OpenSlide(image_path) as slide: # opens the WSI file and reads one patch from it
        for ref in refs:
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

            patches.append(Patch(
                ref=ref,
                img=img_chw,
            ))
    return patches
