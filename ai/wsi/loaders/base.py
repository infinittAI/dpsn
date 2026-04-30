from abc import ABC, abstractmethod
from pathlib import Path

from ai.wsi.handle import WSIHandle
from ai.wsi.patch import Patch
from ai.wsi.patch_ref import PatchRef

class Loader(ABC):
    def __init__(self):
        pass
    
    @staticmethod
    @abstractmethod
    def load_patch(patch_ref: PatchRef) -> Patch:
        pass
    
    @staticmethod
    @abstractmethod
    def open_wsi_handle(img_path: Path) -> WSIHandle:
        pass
