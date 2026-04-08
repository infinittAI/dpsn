import numpy as np
from openslide import OpenSlide

from ai.wsi.patch import Patch
from ai.wsi.patch_ref import PatchRef

def load_patch(ref: PatchRef) -> Patch:
    with OpenSlide(ref.image_path) as slide:
        image = slide.read_region(ref.level0_path, ref.read_level, ref.read_size)
    
    array = np.array(image)

    return Patch(
        ref = ref,
        img = array
    )
