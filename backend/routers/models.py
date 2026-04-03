from fastapi import APIRouter, HTTPException
import json
from schemas import ModelResponse

router = APIRouter()

@router.get("/models", response_model=list[ModelResponse])
def get_models():
    with open("models.json", "r") as f:
        models = json.load(f)
    return [ModelResponse(**model) for model in models]

@router.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: int):
    with open("models.json", "r") as f:
        models = json.load(f)
    for model in models:
        if model["id"] == model_id:
            return model
    raise HTTPException(status_code=404, detail="Model not found")