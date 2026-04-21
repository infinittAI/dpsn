from pathlib import Path

from ai.runtime.worker import Worker
from ai.runtime.task import Task

slide_path = Path("data/GTEX-1117F-0126.svs")

worker = Worker()
result = worker.run(Task(slide_path, slide_path, 1), lambda x: x)
print(result)