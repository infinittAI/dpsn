from abc import ABC, abstractmethod

from ai.wsi.handle import WSIHandle

class Metric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, origin_image_path: str, normalized_image_path: str) -> float:
        pass