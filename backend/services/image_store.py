import uuid
from pathlib import Path
from fastapi import UploadFile

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"

images: dict = {}

# 업로드된 파일을 data/uploads에 저장하고 image_id를 반환
def save_image(file: UploadFile) -> str:
    image_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix if file.filename else ""
    path = UPLOAD_DIR / f"{image_id}{ext}"
    with open(path, "wb") as f:
        f.write(file.file.read())
    images[image_id] = str(path)
    return image_id

# image_id로 저장된 파일 경로를 반환
def get_image_path(image_id: str) -> str:
    return images.get(image_id)
