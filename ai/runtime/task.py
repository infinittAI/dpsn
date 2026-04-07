from dataclasses import dataclass
from pathlib import Path

@dataclass
class Task:
    src_img_path: Path
    target_img_path: Path | None
    model: str

@dataclass
class TaskResult:
    pass