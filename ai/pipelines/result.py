from dataclasses import dataclass

@dataclass
class PipelineResult:
    output_path: str
    scores: dict
