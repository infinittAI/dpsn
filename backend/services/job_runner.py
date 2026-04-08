import uuid
from fastapi import BackgroundTasks

jobs: dict = {}

# model_id와 image_id를 받아 mock 결과(result_image_id, metrics)를 반환하는 임시 worker
# 실제 모델 연결 후에는 AI 파이프라인이 data/results/에 결과 이미지를 저장하고
# 생성한 result_image_id를 직접 반환해야 함
def mock_worker(model_id: int, image_id: str):
    return {
        "result_image_id": str(uuid.uuid4()),
        "ssim": 0.95,
        "psnr": 32.4,
        "fid": 60,
        "model_id": model_id
    }

# job 상태를 running으로 전환하고 worker를 실행해 결과를 jobs dict에 저장
def run_job(job_id: str, model_id: int, image_id: str):
    jobs[job_id]["status"] = "running"

    try:
        result = mock_worker(model_id, image_id)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result_image_id"] = result.pop("result_image_id")
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["result"] = str(e)

# job을 생성하고 background task로 run_job을 스케줄링
def create_job(model_id: int, image_id: str, background_tasks: BackgroundTasks) -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "model_id": model_id,
        "image_id": image_id,
        "result": None
    }
    background_tasks.add_task(run_job, job_id, model_id, image_id)
    return job_id
