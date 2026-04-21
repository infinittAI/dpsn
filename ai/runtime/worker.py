from pathlib import Path

from ai.metrics.ssim import SSIM
from ai.runtime.task import Task, TaskResult, Metrics
from ai.pipelines.base import ModelPipeline
from ai.pipelines.reinhard import Reinhard

# model_id → pipeline 매핑 
PIPELINE_MAP: dict[int, ModelPipeline] = {
    1: Reinhard(),
    # 2: Macenko(),  
    # 3: Vahadane(),  
    # 4: StainGAN(),  
    # 5: StainNet(),  
    # 6: StainSWIN(), 
}

class Worker:
    
    # model_id에 해당하는 pipeline을 실행하고 TaskResult를 반환
    def run(self, task: Task, emit_event) -> TaskResult:
        pipeline = PIPELINE_MAP.get(task.model_id)
        if pipeline is None:
            raise NotImplementedError(f"model_id {task.model_id}에 해당하는 pipeline이 없습니다.")

        result = pipeline.run(task.src_img_path, task.target_img_path, {"ssim": SSIM()})

        metrics = Metrics(
            1.0,
            0.95,
            0.94
        )

        return TaskResult(
            result_img_path=Path(result.output_path),
            metrics=metrics
        )
