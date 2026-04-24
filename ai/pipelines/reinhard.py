from pathlib import Path

import numpy as np
from skimage import color
import time
from collections import defaultdict
from tqdm import tqdm

from ai.metrics.base import Metric
from ai.pipelines.base import ModelPipeline
from ai.pipelines.result import PipelineResult
from ai.samplers.grid_sampler import GridSampler
from ai.samplers.patch_sampler import PatchSampler
from ai.wsi.handle import open_wsi_handle
from ai.wsi.loader import load_patch
from ai.wsi.writer import ZarrWSIWriter

class Reinhard(ModelPipeline):
    target_sampler: PatchSampler

    def __init__(self):
        super().__init__()
        self.patch_sampler = PatchSampler()
        self.grid_sampler = GridSampler(read_level=3)

    def run(
        self,
        src_img_path: Path, 
        target_img_path: Path | None,
        metrics: dict[str, Metric]
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

        scores = dict()
        batch_size = 64
        writer = ZarrWSIWriter(
            "out_img", 
            src_wsi_handle.dim[1], 
            src_wsi_handle.dim[0],
            tile_size = src_refs[0].read_size[0]
        )

        for idx in tqdm(range(0, len(src_refs), batch_size)):
            batch_ref = src_refs[idx:idx + batch_size]
            patches = np.stack([load_patch(ref).img for ref in batch_ref], axis=0)
            
            new_patches = self.transform_image(patches, target_means, target_stds, src_means, src_stds)
            
            for i, ref in enumerate(batch_ref):
                for key, metric in metrics.items():
                    metric.evaluate(patches[i], new_patches[i])

            for i, ref in enumerate(batch_ref):
                writer.write_patch(ref, new_patches[i])
                
        image = writer.get_thumbnail()
        image.save("out_image.png")

        return PipelineResult(output_path="out_image.png")
        
    
    def get_reinhard_stats(self, image: np.ndarray):
        """
        image: RGB image, shape (C, H, W), uint8 or float
        return:
            means: [L_mean, a_mean, b_mean]
            stds:  [L_std,  a_std,  b_std]
        """

        image = image.transpose([1, 2, 0]) # [H, W, C]

        lab = color.rgb2lab(image / 255.0)  # L: [0,100], a/b: roughly [-128,127]

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
        image = image.transpose([0, 2, 3, 1])

        lab = color.rgb2lab(image / 255.0)
        lab = (lab - src_means)/src_stds * target_stds + target_means
        
        image = color.lab2rgb(lab) * 255.0
        image = image.transpose([0, 3, 1, 2])

        return image
