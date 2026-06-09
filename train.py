from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.model.data_loader import NUM_CLASSES, build_dataloader
from src.data.parser import DETECTION_CLASSES  

_VENDOR = Path(__file__).parent / "third_party" / "RT-DETR" / "rtdetrv2_pytorch"

for _k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
    del sys.modules[_k]

sys.path.insert(0, str(_VENDOR))

from src.core import YAMLConfig  
from src.optim import LinearWarmup, ModelEMA  

PROJECT_ROOT = Path(__file__).parent

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "rtdetrv2_bdd100k"
DEFAULT_VENDOR_DIR = _VENDOR
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Model variants: map CLI name → weight filename + config filename.
MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "rtdetrv2-s": {
        "weight_file": "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        "config": "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
    },
    "rtdetrv2-m": {
        "weight_file": "rtdetrv2_r50vd_m_7x_coco_ema.pth",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml",
    },
    "rtdetrv2-l": {
        "weight_file": "rtdetrv2_r50vd_6x_coco_ema.pth",
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune RT-DETRv2 on BDD100K — explicit training loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Root data directory containing bdd100k/",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Checkpoint output directory.",
    )
    parser.add_argument(
        "--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR,
        help="Directory containing the pretrained .pth checkpoint.",
    )
    parser.add_argument(
        "--vendor-dir", type=Path, default=DEFAULT_VENDOR_DIR,
        help="Path to third_party/RT-DETR/rtdetrv2_pytorch.",
    )
    parser.add_argument(
        "--model", default="rtdetrv2-s", choices=sorted(MODEL_REGISTRY),
        help="Model variant.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Base learning rate.")
    parser.add_argument(
        "--train-images", type=int, default=500,
        help="Max training images (subset for quick runs).",
    )
    parser.add_argument(
        "--val-images", type=int, default=100,
        help="Max validation images.",
    )
    parser.add_argument(
        "--img-size", type=int, default=640, help="Square input image size."
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=100,
        help="Number of linear warmup steps.",
    )
    parser.add_argument(
        "--clip-norm", type=float, default=0.1,
        help="Max gradient norm for clipping (0 = disabled).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="'cuda', 'cpu', or a CUDA device index.  Auto-detected when omitted.",
    )
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume a previous run from this checkpoint path.",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable automatic mixed precision even on CUDA.",
    )
    return parser.parse_args()



def _xyxy_to_cxcywh_norm(boxes: torch.Tensor, img_size: int) -> torch.Tensor:
    """Convert [x1, y1, x2, y2] pixel boxes to normalised [cx, cy, w, h].

    RT-DETRv2's criterion and matcher work in the normalised cxcywh space,
    while our DataLoader returns pixel xyxy coordinates resized to img_size.

    Args:
        boxes: Float tensor of shape (N, 4) in [x1, y1, x2, y2] pixel format.
        img_size: Side length the images were resized to (square assumed).

    Returns:
        Float tensor of shape (N, 4) with values in [0, 1].
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5 / img_size
    cy = (y1 + y2) * 0.5 / img_size
    w = (x2 - x1) / img_size
    h = (y2 - y1) / img_size
    return torch.stack([cx, cy, w, h], dim=-1).clamp(0.0, 1.0)


def _prep_targets(
    targets: List[Dict[str, torch.Tensor]],
    img_size: int,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    """Convert a batch of targets from DataLoader format to RT-DETRv2 format.

    DataLoader:   boxes are [x1, y1, x2, y2] pixel coords, labels are 0-indexed.
    RT-DETRv2:    boxes must be normalised [cx, cy, w, h],  labels 0-indexed.
    """
    out = []
    for t in targets:
        boxes = _xyxy_to_cxcywh_norm(t["boxes"].float(), img_size)
        out.append({
            "boxes": boxes.to(device),
            "labels": t["labels"].to(device),
        })
    return out


def _load_pretrained(model: nn.Module, ckpt_path: Path) -> None:
    """Load weights from a COCO checkpoint, ignoring shape-mismatched tensors.

    The COCO checkpoint has a 80-class head; our model is built with 10 classes.
    Keys whose tensor shapes differ (the classification projection layers) are
    silently skipped so the backbone and encoder weights transfer cleanly.

    Args:
        model: The RT-DETRv2 model to load weights into.
        ckpt_path: Path to the .pth checkpoint file.
    """
    state = torch.load(str(ckpt_path), map_location="cpu")

    # The official checkpoint stores weights under 'model' or 'ema.module'.
    src_weights = state.get("ema", {}).get("module") or state.get("model", state)

    model_state = model.state_dict()
    matched, skipped_shape, skipped_missing = {}, [], []

    for k, v in model_state.items():
        if k not in src_weights:
            skipped_missing.append(k)
        elif v.shape != src_weights[k].shape:
            skipped_shape.append(k)
        else:
            matched[k] = src_weights[k]

    model.load_state_dict(matched, strict=False)
    print(f"  Pretrained weights loaded:  {len(matched)} tensors copied")
    print(f"  Shape mismatch (head swap): {len(skipped_shape)} tensors skipped")
    if skipped_missing:
        print(f"  Missing in checkpoint:      {len(skipped_missing)} tensors skipped")


def _build_optimizer(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 1e-4,
) -> torch.optim.AdamW:
    """Build AdamW with two learning-rate groups.
    Args:
        model: The RT-DETRv2 model.
        base_lr: Learning rate for encoder/decoder parameters.
        weight_decay: Default weight decay (applied to non-norm params).

    Returns:
        Configured AdamW optimizer.
    """
    backbone_re = re.compile(r"backbone")
    norm_re = re.compile(r"(?:norm|bn)\.")

    backbone_params, backbone_norm_params = [], []
    head_params, head_norm_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if backbone_re.search(name):
            if norm_re.search(name):
                backbone_norm_params.append(param)
            else:
                backbone_params.append(param)
        else:
            if norm_re.search(name):
                head_norm_params.append(param)
            else:
                head_params.append(param)

    param_groups = [
        # Backbone — lower LR to preserve COCO features.
        {"params": backbone_params,      "lr": base_lr * 0.1, "weight_decay": weight_decay},
        {"params": backbone_norm_params, "lr": base_lr * 0.1, "weight_decay": 0.0},
        # Encoder / decoder head — full LR.
        {"params": head_params,          "lr": base_lr,        "weight_decay": weight_decay},
        {"params": head_norm_params,     "lr": base_lr,        "weight_decay": 0.0},
    ]
    # Drop empty groups so the optimizer doesn't warn about zero-param groups.
    param_groups = [pg for pg in param_groups if pg["params"]]

    total = sum(p.numel() for pg in param_groups for p in pg["params"])
    print(f"  Trainable parameters: {total:,}")
    print(f"  Backbone LR: {base_lr * 0.1:.2e}  |  Head LR: {base_lr:.2e}")

    return torch.optim.AdamW(param_groups, betas=(0.9, 0.999))


def _train_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    warmup: Optional[LinearWarmup],
    ema: Optional[ModelEMA],
    device: torch.device,
    epoch: int,
    img_size: int,
    clip_norm: float,
    print_freq: int = 20,
) -> float:
    """Run one training epoch and return the average total loss.

    Args:
        model: RT-DETRv2 model in training mode.
        criterion: RTDETRCriterionv2 — computes focal + L1 + GIoU losses.
        loader: Training DataLoader.
        optimizer: AdamW optimizer.
        scaler: AMP GradScaler, or None on CPU.
        warmup: Linear warmup scheduler (steps per batch), or None.
        ema: Exponential moving average model, or None.
        device: Target device.
        epoch: Current epoch index (0-based, used for logging).
        img_size: Square image size (needed for box coordinate conversion).
        clip_norm: Max gradient norm; 0 disables clipping.
        print_freq: Log every this many batches.

    Returns:
        Average total loss over all batches in this epoch.
    """
    model.train()
    criterion.train()

    running_loss = 0.0
    n_batches = len(loader)
    t0 = time.time()

    for i, (images, targets) in enumerate(loader):
        # ── Move data to device ──────────────────────────────────────────────
        images = torch.stack(images).to(device)            # (B, 3, H, W)
        targets = _prep_targets(targets, img_size, device) # list of dicts

        global_step = epoch * n_batches + i
        metas = {"epoch": epoch, "step": i, "global_step": global_step}

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type=device.type, cache_enabled=True):
                outputs = model(images, targets=targets)
            with torch.autocast(device_type=device.type, enabled=False):
                loss_dict = criterion(outputs, targets, **metas)
            loss: torch.Tensor = sum(loss_dict.values())

            scaler.scale(loss).backward()
            if clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)
            loss = sum(loss_dict.values())

            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        if warmup is not None:
            warmup.step()

        if ema is not None:
            ema.update(model)

        loss_val = float(loss.detach())
        running_loss += loss_val

        if (i + 1) % print_freq == 0 or (i + 1) == n_batches:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[-1]["lr"]
            detail = "  ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(
                f"  [{epoch + 1}][{i + 1}/{n_batches}]  "
                f"loss: {loss_val:.4f}  {detail}  "
                f"lr: {lr:.2e}  t: {elapsed:.1f}s"
            )

    return running_loss / max(n_batches, 1)



@torch.no_grad()
def _val_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    img_size: int,
) -> float:
    """Compute average validation loss for one epoch (no gradient tracking).

    Args:
        model: RT-DETRv2 model (or EMA shadow model) in eval mode.
        criterion: RTDETRCriterionv2.
        loader: Validation DataLoader.
        device: Target device.
        epoch: Current epoch index (used for metas dict).
        img_size: Square image size.

    Returns:
        Average total validation loss.
    """
    model.eval()
    criterion.eval()

    running_loss = 0.0
    n_batches = len(loader)

    for i, (images, targets) in enumerate(loader):
        images = torch.stack(images).to(device)
        targets = _prep_targets(targets, img_size, device)
        metas = {"epoch": epoch, "step": i, "global_step": 0}

        outputs = model(images, targets=targets)
        loss_dict = criterion(outputs, targets, **metas)
        running_loss += float(sum(loss_dict.values()).detach())

    return running_loss / max(n_batches, 1)



def main() -> None:
    """Build the full RT-DETRv2 pipeline and run training."""
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = (device.type == "cuda") and not args.no_amp

    print("=" * 60)
    print("RT-DETRv2 Fine-tuning — BDD100K")
    print("=" * 60)
    print(f"  Device : {device}  AMP: {use_amp}")
    print(f"  Epochs : {args.epochs}  Batch: {args.batch}  LR: {args.lr:.2e}")

    vendor_dir = Path(args.vendor_dir).resolve()
    if not vendor_dir.is_dir():
        raise FileNotFoundError(
            f"Vendor source not found: {vendor_dir}\n"
            "  Run:  scripts/setup_rtdetrv2.sh"
        )

    spec = MODEL_REGISTRY[args.model]
    ckpt_path = Path(args.weights_dir) / spec["weight_file"]
    if not args.resume and not ckpt_path.exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {ckpt_path}\n"
            f"  Run:  scripts/setup_rtdetrv2.sh {args.model}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Building data loaders...")
    train_loader, class_weights = build_dataloader(
        args.data_dir,
        split="train",
        batch_size=args.batch,
        img_size=args.img_size,
        max_images=args.train_images,
        use_weighted_sampler=True,   # oversample rare classes (train, motor, rider)
        weight_scheme="inverse",
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader, _ = build_dataloader(
        args.data_dir,
        split="val",
        batch_size=args.batch,
        img_size=args.img_size,
        max_images=args.val_images,
        use_weighted_sampler=False,
        shuffle=False,
        num_workers=0,
    )
    print(f"  Train: {len(train_loader.dataset):,} images  ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_loader.dataset):,} images   ({len(val_loader)} batches)")

    print(f"\n[2/5] Building {args.model} model (num_classes={NUM_CLASSES})...")
    cfg_path = vendor_dir / spec["config"]
    cfg = YAMLConfig(
        str(cfg_path),
        num_classes=NUM_CLASSES,
        remap_mscoco_category=False,
        # Disable ImageNet backbone pretraining — the full detector checkpoint
        # already contains those weights; downloading would hit the blocked CDN.
        pretrained=False,
    )
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    print("\n[3/5] Loading pretrained weights...")
    start_epoch = 0
    best_val_loss = math.inf

    if args.resume:
        state = torch.load(str(args.resume), map_location=device)
        model.load_state_dict(state["model"])
        start_epoch = state.get("epoch", 0) + 1
        best_val_loss = state.get("best_val_loss", math.inf)
        print(f"  Resumed from {args.resume}  (epoch {start_epoch})")
    else:
        _load_pretrained(model, ckpt_path)

    print("\n[4/5] Setting up optimiser and scheduler...")
    optimizer = _build_optimizer(model, base_lr=args.lr)

    # MultiStepLR: drop LR by 10× at 67% and 90% of training.
    ms1 = max(1, int(args.epochs * 0.67))
    ms2 = max(ms1 + 1, int(args.epochs * 0.90))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[ms1, ms2], gamma=0.1
    )

    warmup: Optional[LinearWarmup] = None
    if args.warmup_steps > 0:
        warmup = LinearWarmup(lr_scheduler, warmup_duration=args.warmup_steps)
        print(f"  Linear warmup: {args.warmup_steps} steps")
    print(f"  LR milestones: {ms1}, {ms2}  (gamma 0.1)")

    # AMP scaler (no-op on CPU).
    scaler: Optional[torch.cuda.amp.GradScaler] = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("  AMP GradScaler enabled.")

    # EMA shadow model — improves evaluation stability.
    ema = ModelEMA(model, decay=0.9999, warmups=2000)
    print("  EMA model initialised (decay=0.9999).")

    print(f"\n[5/5] Training for {args.epochs} epoch(s)...")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        t_epoch = time.time()

        train_loss = _train_epoch(
            model=model,
            criterion=criterion,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            warmup=warmup,
            ema=ema,
            device=device,
            epoch=epoch,
            img_size=args.img_size,
            clip_norm=args.clip_norm,
        )

        # Step the main LR scheduler after each epoch (warmup handles
        # per-batch steps internally; once finished it stops adjusting LR).
        if warmup is None or warmup.finished():
            lr_scheduler.step()

        # Validation — use EMA model for more stable loss estimate.
        eval_model = ema.module if ema else model
        val_loss = _val_epoch(
            model=eval_model,
            criterion=criterion,
            loader=val_loader,
            device=device,
            epoch=epoch,
            img_size=args.img_size,
        )

        epoch_time = time.time() - t_epoch
        print(
            f"\nEpoch {epoch + 1}/{args.epochs}  "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
            f"time: {epoch_time:.1f}s"
        )

        # Checkpoint: always save the latest state.
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema else None,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
        }
        torch.save(checkpoint, output_dir / "last.pth")

        # Save the best model separately.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / "best.pth")
            print(f"  -> best checkpoint saved (val_loss {val_loss:.4f})")

        print("-" * 60)

    print("\n" + "=" * 60)
    print(f"Training complete.  Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
