from pathlib import Path

from openslide import OpenSlide

from ai.runtime.worker import Worker
from ai.runtime.task import Task

slide_path = Path("data/SCAN algorithm dataset/IMAGES/ADRENAL/Adrenal H&E (20x)/1000240_1.tif")
target_path = Path("data/SCAN algorithm dataset/IMAGES/ADRENAL/TARGET.tif")

worker = Worker()

result = worker.run(Task(slide_path, target_path, 1), lambda x: x)
print(result)