from abc import ABC, abstractmethod

import numpy as np

class Metric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, origin_patch: np.ndarray, normalized_patch: np.ndarray) -> float:
        pass