import os
from dataclasses import dataclass

from ai.metrics.base import Metric
from ai.pipelines.base import ModelPipeline

@dataclass
class Task:
    src_img_path: str | os.PathLike[str]
    target_img_path: str | os.PathLike[str] | None
    model: ModelPipeline
    metrics: list[Metric]
    task_id: str
    output_dir: str

@dataclass
class TaskResult:
    pass