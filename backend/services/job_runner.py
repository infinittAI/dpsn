import uuid

jobs: dict = {}

def create_job(model_id: int) -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "model_id": model_id,
        "result": None
    }
    return job_id