import uuid
from fastapi import BackgroundTasks

jobs: dict = {}

# model_id와 파일을 받아 mock 결과(ssim, psnr)를 반환하는 임시 worker
def mock_worker(model_id: int, file):
    return {
        "ssim": 0.95,
        "psnr": 32.4,
        "model_id": model_id
    }
    
# job 상태를 running으로 전환하고 worker를 실행해 결과를 jobs dict에 저장
def run_job(job_id: str, model_id: int, file):
    jobs[job_id]["status"] = "running"
    
    try:
        result = mock_worker(model_id, file)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["result"] = str(e)

# job을 생성하고 background task로 run_job을 스케줄링
def create_job(model_id: int, file, background_tasks: BackgroundTasks) -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "model_id": model_id,
        "result": None
    }
    background_tasks.add_task(run_job, job_id, model_id, file)
    return job_id