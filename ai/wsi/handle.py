from dataclasses import dataclass
import os
from pathlib import Path

import openslide

from ai.wsi.patch_ref import PatchRef

@dataclass
class WSIHandle:
    image_path: Path
    dim: tuple[int, int]
    mpp: tuple[float, float]
    level_dimensions: tuple
    level_downsamples: tuple
    
    # 해당 레벨에서, Image의 (pos[0], pos[1]) 부터 (pos[0] + dim[0], pos[1] + dim[1]) 까지의 범위를 crop
    def make_ref(self, pos: tuple, level: int, dim: tuple) -> PatchRef:
        level_count = len(self.level_dimensions)

        if level >= level_count:
            raise ValueError(
                f"Level: {level} must be less than maximum level: {level_count}"
            )
        
        # Image의 범위가 유효한지 확인
        img_width, img_height = self.level_dimensions[level]

        if not (0 <= pos[0] <= img_width - dim[0]):
            raise ValueError(
                f"Invalid width range: {(pos[0], pos[0] + dim[0])} for {(0, img_width)}"
            )
        
        if not (0 <= pos[1] <= img_height - dim[1]):
            raise ValueError(
                f"Invalid height range: {(pos[1], pos[1] + dim[1])} for {(0, img_height)}"
            )
        
        return PatchRef()


# 사용자로부터 Image path를 전달받아서, 
# 다양한 wsi format(.tif ...)에 따라 WSIHandle을 구성하고 리턴
def open_wsi_handle(image_path: str | os.PathLike[str]) -> WSIHandle:
    image_path = Path(image_path)
    
    # openslide에서 지원하는 형식인지 확인
    if openslide.OpenSlide.detect_format(image_path):
        with openslide.OpenSlide(image_path) as slide:
            width, height = slide.dimensions
            props = slide.properties
            mpp_x = props.get(openslide.PROPERTY_NAME_MPP_X)
            mpp_y = props.get(openslide.PROPERTY_NAME_MPP_Y)

            # mpp_x와 mpp_y가 항상 제공되지 않을 수 있다.
            # 이때 우선 -1을 저장하는데, 추후 처리가 필요할 듯 하다.
            mpp_x = float(mpp_x if mpp_x is not None else "-1")
            mpp_y = float(mpp_y if mpp_y is not None else "-1")
            
            level_dimensions = slide.level_dimensions # 각 레벨별 Image Size
            level_downsamples = slide.level_downsamples # 레벨 0에 비해 얼마나 downsample 되었는지
        
            return WSIHandle(
                image_path = image_path,
                dim = (width, height),
                mpp = (mpp_x, mpp_y),
                level_dimensions = level_dimensions,
                level_downsamples = level_downsamples,
            )
    else:
        raise NotImplementedError
