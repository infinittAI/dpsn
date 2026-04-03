from pydantic import BaseModel

class ModelResponse(BaseModel):
    id: int
    name: str
    category: str
    description: str

class JobResponse(BaseModel):
    job_id: str
    
class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending" / "running" / "done" / "failed"