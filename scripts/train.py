"""
NatyaVeda — Training Script
Usage:
  python scripts/train.py --config config/config.yaml --model danceformer-large
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="NatyaVeda DanceFormer Training")
    parser.add_argument("--config",   default="config/config.yaml")
    parser.add_argument("--data",     default="data/processed")
    parser.add_argument("--model",    default="danceformer-large",
                        choices=["danceformer-small", "danceformer-base", "danceformer-large"])
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--batch",    type=int, default=None)
    parser.add_argument("--lr",       type=float, default=None)
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",   default="weights")
    parser.add_argument("--resume",   default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override with CLI args
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch:
        config["training"]["batch_size"] = args.batch
    if args.lr:
        config["training"]["optimizer"]["lr"] = args.lr
    if args.no_wandb:
        config["training"]["logging"]["use_wandb"] = False

    logger.info("=" * 60)
    logger.info("  NatyaVeda Analyzer — Training")
    logger.info("=" * 60)
    logger.info(f"  Model    : {args.model}")
    logger.info(f"  Data     : {args.data}")
    logger.info(f"  Device   : {args.device}")
    logger.info(f"  Epochs   : {config['training']['epochs']}")
    logger.info(f"  Batch    : {config['training']['batch_size']}")
    logger.info(f"  LR       : {config['training']['optimizer']['lr']}")

    # Import here to avoid slow startup for --help
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.danceformer import MODEL_REGISTRY
    from src.training.trainer import DanceDataset, Trainer

    # Dataset
    data_dir = Path(args.data)
    ds_cfg = config.get("dataset", {})
    clip_len = ds_cfg.get("clip_length_frames", 64)
    clip_stride = ds_cfg.get("clip_stride_frames", 32)

    train_ds = DanceDataset(data_dir, split="train", clip_length=clip_len,
                            clip_stride=clip_stride, augment=True)
    val_ds   = DanceDataset(data_dir, split="val",   clip_length=clip_len,
                            clip_stride=16, augment=False)

    if len(train_ds) == 0:
        logger.error("No training data found. Run scripts/extract_features.py first.")
        sys.exit(1)

    # Model
    model = MODEL_REGISTRY[args.model]()
    logger.info(f"  Params   : {model.count_parameters():,}")

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])
        logger.info(f"  Resumed  : {args.resume}")

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        config=config,
        device=args.device,
        output_dir=args.output,
    )
    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
