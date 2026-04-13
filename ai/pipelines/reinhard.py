from pathlib import Path

import numpy as np
from skimage import color

from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.samplers.grid_sampler import GridSampler
from ai.samplers.patch_sampler import PatchSampler
from ai.wsi.handle import open_wsi_handle
from ai.wsi.loader import load_patch

class Reinhard(ModelPipeline):
    target_sampler: PatchSampler

    def __init__(self):
        super().__init__()
        self.patch_sampler = PatchSampler()
        self.grid_sampler = GridSampler()

    def run(
        self,
        src_img_path: Path, 
        target_img_path: Path | None
    ) -> PipelineResult:
        
        if target_img_path is None:
            raise ValueError("Reinhard needs a target image.")
        
        src_wsi_handle = open_wsi_handle(src_img_path)
        target_wsi_handle = open_wsi_handle(target_img_path)

        src_ref = self.patch_sampler.sample(src_wsi_handle, max_patches=1)[0]
        src_patch = load_patch(src_ref)
        target_ref = self.patch_sampler.sample(target_wsi_handle, max_patches=1)[0]
        target_patch = load_patch(target_ref)

        target_means, target_stds = self.get_reinhard_stats(target_patch.img)
        src_means, src_stds = self.get_reinhard_stats(src_patch.img)

        src_refs = self.grid_sampler.sample(src_wsi_handle)

        total_images = []
        for ref in src_refs:
            patch = load_patch(ref)
            new_img = patch.img.copy()
            new_img = self.transform_image(new_img, target_means, target_stds, src_means, src_stds)
            total_images.append(new_img)

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
    
    def transform_image(
        self, 
        image: np.ndarray, 
        target_means: np.ndarray,
        target_stds: np.ndarray,
        src_means: np.ndarray,
        src_stds: np.ndarray, 
    ) -> np.ndarray:
        image = image.transpose([1, 2, 0])

        lab = color.rgb2lab(image)
        lab = (lab - src_means)/src_stds * target_stds + target_means
        
        image = color.lab2rgb(image)
        image = image.transpose([2, 0, 1])

        return image
