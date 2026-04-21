from skimage.metrics import structural_similarity as ssim

from ai.metrics.base import Metric
from ai.samplers.grid_sampler import GridSampler
from ai.wsi.handle import open_wsi_handle
from ai.wsi.loader import load_patch

class SSIM(Metric):
    def __init__(self):
        pass

    def evaluate(
        self,
        origin_image_path: str,
        normalized_image_path: str
    ) -> float:
        sampler = GridSampler()
        origin_image = open_wsi_handle(origin_image_path)
        normalized_image = open_wsi_handle(normalized_image_path)
        origin_refs = sampler.sample(origin_image)
        normalized_refs = sampler.sample(normalized_image)

        total_scores = []
        iter = 0
        for ref1, ref2 in zip(origin_refs, normalized_refs):
            patch1, patch2 = load_patch(ref1), load_patch(ref2)
            score = ssim(patch1.img, patch2.img, data_range=255, channel_axis=0)
            total_scores.append(score)
            if iter > 1000:
                break
            iter += 1
        
        return sum(total_scores) / len(total_scores)
