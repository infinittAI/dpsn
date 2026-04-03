from fastapi import FastAPI
from routers import models, jobs

app = FastAPI()

app.include_router(models.router, tags=["models"])
app.include_router(jobs.router, tags=["jobs"])