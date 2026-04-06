from abc import ABC, abstractmethod

from ai.pipelines.result import PipelineResult
from ai.wsi.handle import WSIHandle

class ModelPipeline(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def run(self, src_img_path: str, target_img_path: str) -> PipelineResult: ...
    