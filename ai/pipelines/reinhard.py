import os

import numpy as np
from skimage import color

from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.wsi.handle import open_wsi_handle
from ai.wsi.loader import load_patch

class Reinhard(ModelPipeline):
    def __init__(self):
        super().__init__()

    def run(
        self,
        src_img_path: str | os.PathLike[str], 
        target_img_path: str | os.PathLike[str] | None
    ) -> PipelineResult:
        
        if target_img_path is None:
            raise ValueError("Reinhard needs a target image.")
        
        src_wsi_handle = open_wsi_handle(src_img_path)
        target_wsi_handle = open_wsi_handle(target_img_path)

        target_ref = target_wsi_handle.make_ref((0, 0), 0, target_wsi_handle.dim)
        target_patch = load_patch(target_ref)

        target_means, target_stds = self.get_reinhard_stats(target_patch.img)

        raise NotImplementedError
    
    def get_reinhard_stats(self, image: np.ndarray):
        """
        image: RGB image, shape (C, H, W), uint8 or float
        return:
            means: [L_mean, a_mean, b_mean]
            stds:  [L_std,  a_std,  b_std]
        """
        if image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32) / 255.0

        image = image.transpose([1, 2, 0]) # [H, W, C]

        lab = color.rgb2lab(image)  # L: [0,100], a/b: roughly [-128,127]

        means = lab.reshape(-1, 3).mean(axis=0)
        stds = lab.reshape(-1, 3).std(axis=0)

        return means, stds
