from abc import ABC, abstractmethod
import os

from ai.pipelines.result import PipelineResult

class ModelPipeline(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def run(
        self, 
        src_img_path: str | os.PathLike[str], 
        target_img_path: str | os.PathLike[str] | None
    ) -> PipelineResult: ...
