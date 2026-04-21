import numpy as np
from skimage.metrics import structural_similarity as ssim

from ai.metrics.base import Metric

class SSIM(Metric):
    def __init__(self):
        pass

    def evaluate(
        self,
        origin_patch: np.ndarray,
        normalized_patch: np.ndarray
    ) -> float:
        score = ssim(origin_patch, normalized_patch, data_range=255, channel_axis=0)
        
        return score
