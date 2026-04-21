from pathlib import Path

from ai.wsi.handle import open_wsi_handle
from ai.samplers.patch_sampler import PatchSampler
from ai.wsi.loader import load_patch

# slide_path = Path("../patient_178/patient_178_node_1.tif") #relative to dpsn directory
slide_path = Path("data/GTEX-1117F-0126.svs")

# Run command (run from dpsn directory): python -m ai.wsi.smoke_test


handle = open_wsi_handle(slide_path)

sampler = PatchSampler(
    patch_size=256,
    read_level=0,
    strict_mpp_check=False,
    result_dir="result_smoke",
)

refs = sampler.sample(
    handle,
    mode="inference",
    save_debug=True,
)

print("patch refs:", len(refs))

first_patch = load_patch(refs[0])
print("patch shape:", first_patch.img.shape)
print("patch dtype:", first_patch.img.dtype)
print("patch ref:", first_patch.ref)
