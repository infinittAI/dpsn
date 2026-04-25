from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ai.datasets.paired_aligned_dataset import PairedAlignedImageDataset
from dpsn.ai.models.stainnet.stainnet_model import StainNet


DEFAULT_DATASET_ROOT = Path("/home/snu_2026/dpsn/datasets")


@dataclass(slots=True)
class StainNetTrainingConfig:
    train_source_dir: Path = DEFAULT_DATASET_ROOT / "train" / "source"
    train_target_dir: Path = DEFAULT_DATASET_ROOT / "train" / "target"
    val_source_dir: Path | None = None
    val_target_dir: Path | None = None
    checkpoints_dir: Path = Path("checkpoints")
    image_size: int = 256
    input_nc: int = 3
    output_nc: int = 3
    channels: int = 32
    n_layer: int = 3
    kernel_size: int = 1
    batch_size: int = 8
    num_workers: int = 0
    lr: float = 0.01
    epochs: int = 10
    device: str = "auto"
    experiment_name: str = "stainnet"


def create_model(config: StainNetTrainingConfig) -> StainNet:
    return StainNet(
        input_nc=config.input_nc,
        output_nc=config.output_nc,
        n_layer=config.n_layer,
        n_channel=config.channels,
        kernel_size=config.kernel_size,
    )


def create_dataloader(
    source_dir: Path,
    target_dir: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = PairedAlignedImageDataset(
        source_dir=source_dir,
        target_dir=target_dir,
        image_size=image_size,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for source, target, _ in dataloader:
        source = source.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)

        optimizer.zero_grad()
        output = model(source)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.inference_mode():
        for source, target, _ in dataloader:
            source = source.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            output = model(source)
            loss = loss_fn(output, target)
            total_loss += float(loss.item())
            total_batches += 1

    return total_loss / max(total_batches, 1)


def save_checkpoint(
    model: nn.Module,
    config: StainNetTrainingConfig,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "experiment_name": config.experiment_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
        },
        path,
    )


def train(config: StainNetTrainingConfig) -> Path:
    device = select_device(config.device)
    train_loader = create_dataloader(
        source_dir=config.train_source_dir,
        target_dir=config.train_target_dir,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
    )

    val_loader = None
    if config.val_source_dir is not None and config.val_target_dir is not None:
        val_loader = create_dataloader(
            source_dir=config.val_source_dir,
            target_dir=config.val_target_dir,
            image_size=config.image_size,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=False,
        )

    model = create_model(config).to(device)
    optimizer = SGD(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    loss_fn = nn.L1Loss()

    latest_checkpoint_path = (
        config.checkpoints_dir / f"{config.experiment_name}_latest.pth"
    )

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
        )

        if val_loader is not None:
            val_loss = evaluate(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
            )
            print(
                f"epoch {epoch}/{config.epochs} - train_l1={train_loss:.6f} val_l1={val_loss:.6f}"
            )
        else:
            print(f"epoch {epoch}/{config.epochs} - train_l1={train_loss:.6f}")

        scheduler.step()
        save_checkpoint(
            model=model,
            config=config,
            epoch=epoch,
            optimizer=optimizer,
            path=latest_checkpoint_path,
        )

    return latest_checkpoint_path


def select_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train StainNet on paired aligned images.")
    parser.add_argument("--train-source-dir", type=Path, default=DEFAULT_DATASET_ROOT / "train" / "source")
    parser.add_argument("--train-target-dir", type=Path, default=DEFAULT_DATASET_ROOT / "train" / "target")
    parser.add_argument("--val-source-dir", type=Path, default=None)
    parser.add_argument("--val-target-dir", type=Path, default=None)
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--experiment-name", type=str, default="stainnet")
    parser.add_argument("--input-nc", type=int, default=3)
    parser.add_argument("--output-nc", type=int, default=3)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--n-layer", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=1)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = StainNetTrainingConfig(
        train_source_dir=args.train_source_dir,
        train_target_dir=args.train_target_dir,
        val_source_dir=args.val_source_dir,
        val_target_dir=args.val_target_dir,
        checkpoints_dir=args.checkpoints_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        experiment_name=args.experiment_name,
        input_nc=args.input_nc,
        output_nc=args.output_nc,
        channels=args.channels,
        n_layer=args.n_layer,
        kernel_size=args.kernel_size,
    )
    checkpoint_path = train(config)
    print(f"Saved checkpoint to: {checkpoint_path}")


if __name__ == "__main__":
    main()
