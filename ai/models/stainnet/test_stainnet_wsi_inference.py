from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import tifffile
import torch

from ai.models.stainnet.stainnet_model import StainNet
from ai.pipelines.stainnet import StainNetInferenceConfig, StainNetPipeline
from ai.wsi.handle import open_wsi_handle


def create_temp_checkpoint(path: Path) -> Path:
    model = StainNet(
        input_nc=3,
        output_nc=3,
        n_layer=3,
        n_channel=32,
        kernel_size=1,
    )
    torch.save(model.state_dict(), path)
    return path


def run_inference_smoke_test(
    input_wsi: Path,
    checkpoint_path: Path | None,
    read_level: int,
    output_dir: Path,
    batch_size: int,
    allow_single_level: bool,
) -> Path:
    if not input_wsi.is_file():
        raise FileNotFoundError(f"Input WSI not found: {input_wsi}")

    wsi_handle = open_wsi_handle(input_wsi)
    if len(wsi_handle.level_dimensions) < 2 and not allow_single_level:
        raise ValueError(
            f"Input WSI must be pyramidal for this smoke test, got {len(wsi_handle.level_dimensions)} level(s)."
        )

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if checkpoint_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        checkpoint_path = create_temp_checkpoint(Path(temp_dir.name) / "stainnet_temp.pth")

    config = StainNetInferenceConfig(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        read_level=read_level,
        batch_size=batch_size,
    )
    pipeline = StainNetPipeline(config=config)
    result = pipeline.run(input_wsi)

    output_path = Path(result.output_path)
    if not output_path.exists():
        raise AssertionError(f"Output file was not created: {output_path}")

    expected_w, expected_h = wsi_handle.level_dimensions[read_level]
    with tifffile.TiffFile(output_path) as tif:
        page = tif.pages[0]
        output_shape = page.shape

    if output_shape != (expected_h, expected_w, 3):
        raise AssertionError(
            f"Unexpected output shape {output_shape}, expected {(expected_h, expected_w, 3)}"
        )

    print("WSI inference smoke test passed.")
    print(f"input_wsi: {input_wsi}")
    print(f"output_path: {output_path}")
    print(f"read_level: {read_level}")
    print(f"output_shape: {output_shape}")

    if temp_dir is not None:
        temp_dir.cleanup()

    return output_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a StainNet WSI inference smoke test.")
    parser.add_argument("--input-wsi", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--read-level", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("result_stainnet_smoke"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--allow-single-level", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_inference_smoke_test(
        input_wsi=args.input_wsi,
        checkpoint_path=args.checkpoint_path,
        read_level=args.read_level,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        allow_single_level=args.allow_single_level,
    )


if __name__ == "__main__":
    main()
