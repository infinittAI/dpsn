import numpy as np
from skimage.metrics import structural_similarity as ssim

from ai.metrics.base import Metric
from ai.samplers.grid_sampler import GridSampler

class SSIM(Metric):
    def __init__(
        self,
        patch_size: int = 256,
        stride: int | None = None,
        read_level: int = 0,
    ) -> None:
        self.sampler = GridSampler(
            patch_size=patch_size,
            stride=stride,
            read_level=read_level,
        )

    def evaluate(
        self,
        origin_patch: np.ndarray,
        normalized_patch: np.ndarray
    ) -> float:
        
        score = ssim(origin_patch, normalized_patch, channel_axis=1, data_range=255)
        return score
