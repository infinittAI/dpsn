
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np

from ai.metrics.base import Metric



class FID(Metric):
    def __init__(self):
        super().__init__()

    def evaluate(self, origin_patch: np.ndarray, normalized_patch: np.ndarray) -> float:
        fid = FrechetInceptionDistance(feature=2048, normalize=False)

        target_imgs = torch.from_numpy(origin_patch).to(dtype=torch.uint8)
        norm_imgs = torch.from_numpy(normalized_patch).to(dtype=torch.uint8)

        fid.update(target_imgs, real=True)
        fid.update(norm_imgs, real=False)

        return float(fid.compute().item())

