from pathlib import Path

from ai.pipelines.stainnet import StainNetInferenceConfig, StainNetPipeline

input_wsi = Path("/mnt/Disk1/dpsn_datasets/camelyon16/normal_074.tif")
checkpoint = Path("ai/checkpoints/stainnet/stainnet_aperio_to_hamamatsu_latest.pth")

config = StainNetInferenceConfig(
    checkpoint_path=checkpoint,
    output_dir=Path("result_stainnet"),
    read_level=0,
    batch_size=8,
    verbose=True,
    log_every_batches=5,
)

pipeline = StainNetPipeline(config=config)
result = pipeline.run(input_wsi)

print("output_path:", result.output_path)
