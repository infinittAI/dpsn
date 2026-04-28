from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from backend.routers import models, jobs, images
from routers import models, jobs, images

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router, tags=["models"])
app.include_router(jobs.router, tags=["jobs"])
app.include_router(images.router, tags=["images"])