from dataclasses import dataclass
from pathlib import Path

@dataclass
class PipelineResult:
    output_path: str | Path
    scores: dict
