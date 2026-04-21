from ai.wsi.handle import WSIHandle
from ai.wsi.patch_ref import PatchRef

class GridSampler:
    def __init__(
        self, 
        patch_size: int = 256,
        stride: int | None = None,
        read_level: int = 0,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.read_level = read_level
    
    def sample(
        self, 
        wsi_handle: WSIHandle
    ):
        size = wsi_handle.level_dimensions[self.read_level]
        stride = self.patch_size if self.stride is None else self.stride

        patch_refs = []
        for i in range(0, size[0] - stride + 1, stride):
            for j in range(0, size[1] - stride + 1, stride):
                patch_ref = wsi_handle.make_ref(
                    pos = (i, j),
                    level = self.read_level,
                    dim = (self.patch_size, self.patch_size)
                )
                patch_refs.append(patch_ref)

        return patch_refs
