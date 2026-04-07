from fastapi import APIRouter, HTTPException
import json
from schemas import ModelResponse
from pathlib import Path

router = APIRouter()
BASE_DIR = Path(__file__).resolve().parent.parent

# models.json을 읽어 전체 모델 목록을 반환
@router.get("/models", response_model=list[ModelResponse])
def get_models():
    with open(BASE_DIR / "models.json", "r") as f:
        models = json.load(f)
    return [ModelResponse(**model) for model in models]

# model_id로 특정 모델을 조회하고 없으면 404를 반환
@router.get("/models/{model_id}", response_model=ModelResponse)
def get_model(model_id: int):
    with open("models.json", "r") as f:
        models = json.load(f)
    for model in models:
        if model["id"] == model_id:
            return model
    raise HTTPException(status_code=404, detail="Model not found")