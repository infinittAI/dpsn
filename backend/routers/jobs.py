from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from schemas import JobResponse, JobStatusResponse
from services import job_runner

router = APIRouter()

@router.post("/jobs", response_model=list[JobResponse])
async def create_job(
    image: UploadFile = File(...),
    model_ids: str = Form(...)
):
    model_id_list = [int(x) for x in model_ids.split(",")]    
    
    return [JobResponse(job_id = job_runner.create_job(mid)) for mid in model_id_list]

@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    if job_id not in job_runner.jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = job_runner.jobs[job_id]
    return JobStatusResponse(job_id=job_id, status=job["status"])

@router.get("/jobs/{job_id}/results")
async def get_job_results(job_id: int):
    # TODO: 실제 결과 반환 (AI 모델 연결 후 구현)
    return {
        "job_id": job_id,
        "status": "done",
        "metrics": {
            "ssim": 0.0,
            "psnr": 0.0
        },
        "result_image_url": ""
    }