from fastapi import FastAPI
from routers import models, jobs, images

app = FastAPI()

app.include_router(models.router, tags=["models"])
app.include_router(jobs.router, tags=["jobs"])
app.include_router(images.router, tags=["images"])