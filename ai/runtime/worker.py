from datetime import datetime
import logging
from pathlib import Path
from typing import Callable

from ai.metrics.ssim import SSIM
from ai.runtime.task import Task, TaskResult, Metrics
from ai.pipelines.base import ModelPipeline
from ai.pipelines.reinhard import Reinhard

# model_id → pipeline 매핑 
PIPELINE_MAP: dict[int, Callable[..., ModelPipeline]] = {
    1: lambda params: Reinhard(**params),
    # 2: Macenko(),  
    # 3: Vahadane(),  
    # 4: StainGAN(),  
    # 5: StainNet(),  
    # 6: StainSWIN(), 
}

class Worker:

    def _build_logger(self, task_id: int) -> logging.Logger:
        logger_name = f"Task {task_id:04d}:{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        # avoid duplicated handlers if recreated
        if logger.handlers:
            logger.handlers.clear()

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f"result/{timestamp}.txt", encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger
        
    
    # model_id에 해당하는 pipeline을 실행하고 TaskResult를 반환
    def run(self, task: Task, emit_event) -> TaskResult:
        logger = self._build_logger(id(self))
        pipeline = PIPELINE_MAP.get(task.model_id, lambda params: None)({"logger": logger})
        if pipeline is None:
            raise NotImplementedError(f"model_id {task.model_id}에 해당하는 pipeline이 없습니다.")

        result = pipeline.run(
            task.src_img_path, 
            task.target_img_path, 
            {"ssim": SSIM()},
        )

        metrics = Metrics(
            1.0,
            0.95,
            0.94
        )

        return TaskResult(
            result_img_path=Path(result.output_path),
            metrics=metrics
        )
