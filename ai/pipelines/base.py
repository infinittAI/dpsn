from abc import ABC, abstractmethod
from pathlib import Path

from ai.pipelines.result import PipelineResult

class ModelPipeline(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def run(
        self, 
        src_img_path: Path, 
        target_img_path: Path | None
    ) -> PipelineResult: ...
