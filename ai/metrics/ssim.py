from skimage.metrics import structural_similarity as ssim

from ai.metrics.base import Metric
from ai.samplers.grid_sampler import GridSampler
from ai.wsi.handle import WSIHandle
from ai.wsi.loader import load_patch

class SSIM(Metric):
    def __init__(self):
        pass

    def evaluate(
        self,
        origin_image: WSIHandle,
        normalized_image: WSIHandle
    ) -> float:
        sampler = GridSampler()
        origin_refs = sampler.sample(origin_image)
        normalized_refs = sampler.sample(normalized_image)

        total_scores = []
        for ref1, ref2 in zip(origin_refs, normalized_refs):
            patch1, patch2 = load_patch(ref1), load_patch(ref2)
            score = ssim(patch1.img, patch2.img, data_range=255, channel_axis=-1)[0]
            total_scores.append(score)
        
        return sum(total_scores) / len(total_scores)
