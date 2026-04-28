import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

from ai.metrics.base import Metric

class PSNR(Metric):
    def __init__(self):
        super().__init__()

    def evaluate(
        self,
        origin_patch: np.ndarray,
        normalized_patch: np.ndarray
    ) -> float:
        score = psnr(origin_patch, normalized_patch, data_range=255)
        
        return score