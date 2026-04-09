from dataclasses import dataclass
from pathlib import Path

@dataclass
class Metrics:
    ssim: float
    psnr: float
    fid: float

@dataclass
class Task:
    src_img_path: Path
    target_img_path: Path | None
    model_id: int  # models.json의 id와 대응

@dataclass
class TaskResult:
    result_img_path: Path
    metrics: Metrics
