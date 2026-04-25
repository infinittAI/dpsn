from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

from ai.metrics.ssim import SSIM
from ai.pipelines.base import ModelPipeline
from ai.runtime.task import Metrics, Task, TaskResult


class WorkerError(RuntimeError):
    """Base class for worker errors."""


class UnknownModelError(WorkerError):
    """Raised when a task asks for a model_id that is not available."""


class InvalidPipelineResultError(WorkerError):
    """Raised when a pipeline does not return the expected output path."""


PIPELINE_MAP: dict[int, str] = {
    1: "ai.pipelines.reinhard:Reinhard",
    5: "ai.pipelines.stainnet:StainNetPipeline",
}


class Worker:
    """Simple runtime coordinator for one normalization task."""

    def run(self, task: Task, emit_event=None) -> TaskResult:
        pipeline = self._create_pipeline(task.model_id)
        pipeline_result = pipeline.run(
            task.src_img_path, 
            task.target_img_path,
            {"ssim": SSIM()}
        )

        result_img_path = self._get_result_img_path(pipeline_result)

        # Metrics are still placeholder-level in the current project.
        metrics = Metrics(
            ssim=float("nan"),
            psnr=float("nan"),
            fid=float("nan"),
        )

        return TaskResult(
            result_img_path=result_img_path,
            metrics=metrics,
        )

    def _create_pipeline(self, model_id: int) -> ModelPipeline:
        pipeline_path = PIPELINE_MAP.get(model_id)
        if pipeline_path is None:
            raise UnknownModelError(
                f"model_id {model_id} does not have a registered pipeline."
            )

        module_path, class_name = pipeline_path.split(":", maxsplit=1)
        module = import_module(module_path)
        pipeline_class = getattr(module, class_name)
        return pipeline_class()

    def _get_result_img_path(self, pipeline_result: Any) -> Path:
        output_path = getattr(pipeline_result, "output_path", None)
        if not output_path:
            raise InvalidPipelineResultError(
                "Pipeline result must contain a non-empty output_path."
            )

        return Path(output_path)
