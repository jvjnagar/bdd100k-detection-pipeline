from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .class_map import CLASS_TO_ID, ID_TO_CLASS
from ..data.parser import DETECTION_CLASSES, ImageAnnotation, load_split

NUM_CLASSES = len(DETECTION_CLASSES)

# A target is a dict of tensors (torchvision detection convention).
Target = Dict[str, torch.Tensor]


class BDD100KDetectionDataset(Dataset):
    """PyTorch ``Dataset`` for BDD100K object detection.

    Each sample is an ``(image, target)`` pair where:

    * ``image`` is a ``float32`` tensor of shape ``(3, H, W)`` with values in
      ``[0, 1]`` (RGB, resized to a square ``img_size``).
    * ``target`` is a dict with keys ``boxes`` ``(N, 4)`` in ``[x1, y1, x2, y2]``
      pixel coordinates of the resized image, ``labels`` ``(N,)`` int64 class
      ids, ``image_id`` ``(1,)``, ``area`` ``(N,)``, ``iscrowd`` ``(N,)`` and
      ``orig_size`` ``(2,)`` holding the original ``(height, width)``.

    Args:
        data_dir: Root data directory containing ``bdd100k/``.
        split: One of ``"train"``, ``"val"`` or ``"test"``.
        img_size: Side length the images/boxes are resized to (square).
        max_images: Optional cap on the number of images (for quick subsets).
        filter_empty: Drop images that contain no detection-class objects.
        transforms: Optional callable ``(image, target) -> (image, target)``
            applied after loading (e.g. augmentations).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        img_size: int = 640,
        max_images: Optional[int] = None,
        filter_empty: bool = True,
        transforms: Optional[Callable[[torch.Tensor, Target], Tuple[torch.Tensor, Target]]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = int(img_size)
        self.transforms = transforms
        self.class_to_id = dict(CLASS_TO_ID)
        self.id_to_class = dict(ID_TO_CLASS)

        annotations = load_split(str(self.data_dir), split) or []
        image_dir = self._resolve_image_dir()

        self.samples: List[Tuple[Path, ImageAnnotation]] = []
        for ann in annotations:
            if filter_empty and not ann.bboxes:
                continue
            img_path = self._resolve_image_path(image_dir, ann.filename)
            if img_path is None:
                continue
            self.samples.append((img_path, ann))
            if max_images is not None and len(self.samples) >= max_images:
                break

    def _resolve_image_dir(self) -> Path:
        """Return the first existing images directory for the split."""
        candidates = [
            self.data_dir / "bdd100k" / "images" / "100k" / self.split,
            self.data_dir / "images" / "100k" / self.split,
            self.data_dir / "images" / self.split,
        ]
        for cand in candidates:
            if cand.is_dir():
                return cand
        # Fall back to the canonical location even if absent (handled per-item).
        return candidates[0]

    @staticmethod
    def _resolve_image_path(image_dir: Path, filename: str) -> Optional[Path]:
        """Resolve an annotation filename to an existing image path."""
        stem = Path(filename).stem
        candidates = [image_dir / filename, image_dir / f"{stem}.jpg", image_dir / f"{stem}.png"]
        for cand in candidates:
            if cand.exists():
                return cand
        return None

    def __len__(self) -> int:
        """Number of usable samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Target]:
        """Load and return the ``(image, target)`` pair at ``idx``."""
        img_path, ann = self.samples[idx]

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = rgb.shape[:2]

        # Resize image to a square and scale boxes by the same factors.
        resized = cv2.resize(
            rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR
        )
        scale_x = self.img_size / float(orig_w)
        scale_y = self.img_size / float(orig_h)

        boxes: List[List[float]] = []
        labels: List[int] = []
        for bbox in ann.bboxes:
            x1 = max(0.0, min(bbox.x1 * scale_x, self.img_size - 1))
            y1 = max(0.0, min(bbox.y1 * scale_y, self.img_size - 1))
            x2 = max(0.0, min(bbox.x2 * scale_x, self.img_size))
            y2 = max(0.0, min(bbox.y2 * scale_y, self.img_size))
            if x2 <= x1 or y2 <= y1:
                continue  # skip degenerate boxes after scaling
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_to_id[bbox.category])

        # CHW float image in [0, 1].
        image = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float() / 255.0

        if boxes:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)

        target: Target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((labels_t.shape[0],), dtype=torch.int64),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def class_counts(self) -> torch.Tensor:
        """Return a ``(NUM_CLASSES,)`` tensor of object counts per class."""
        counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        for _, ann in self.samples:
            for bbox in ann.bboxes:
                counts[self.class_to_id[bbox.category]] += 1
        return counts

def compute_class_weights(
    dataset: BDD100KDetectionDataset,
    scheme: str = "inverse",
    beta: float = 0.999,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute per-class **weights** to counter class imbalance.

    Args:
        dataset: A :class:`BDD100KDetectionDataset` to count classes from.
        scheme: Weighting scheme:
            * ``"inverse"`` — ``total / (num_classes * count_c)``.
            * ``"inverse_sqrt"`` — inverse of the square-root frequency.
            * ``"effective"`` — effective-number weighting
              ``(1 - beta) / (1 - beta**count_c)`` (Cui et al., 2019).
        beta: Hyper-parameter for the ``"effective"`` scheme.
        normalize: If ``True``, scale weights so their mean is 1.

    Returns:
        A ``(NUM_CLASSES,)`` float tensor of class weights.
    """
    counts = dataset.class_counts().clamp(min=1.0)
    total = counts.sum()

    if scheme == "inverse":
        weights = total / (NUM_CLASSES * counts)
    elif scheme == "inverse_sqrt":
        weights = torch.sqrt(total) / torch.sqrt(NUM_CLASSES * counts)
    elif scheme == "effective":
        eff_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / eff_num
    else:
        raise ValueError(f"Unknown weighting scheme: {scheme!r}")

    if normalize:
        weights = weights * (NUM_CLASSES / weights.sum())
    return weights.float()


def compute_sample_weights(
    dataset: BDD100KDetectionDataset,
    class_weights: Optional[torch.Tensor] = None,
    aggregation: str = "max",
) -> torch.Tensor:
    """Compute a per-image sampling weight from the classes it contains.

    Images that contain rare classes receive a higher weight, so a
    :class:`~torch.utils.data.WeightedRandomSampler` oversamples them.

    Args:
        dataset: The dataset to weight.
        class_weights: Optional precomputed class weights; computed if ``None``.
        aggregation: How to reduce the per-object class weights of an image —
            ``"max"`` (surface the rarest class) or ``"mean"``.

    Returns:
        A ``(len(dataset),)`` float tensor of per-sample weights.
    """
    if class_weights is None:
        class_weights = compute_class_weights(dataset)

    fallback = float(class_weights.min())
    sample_weights = torch.empty(len(dataset), dtype=torch.float32)
    for i, (_, ann) in enumerate(dataset.samples):
        if not ann.bboxes:
            sample_weights[i] = fallback
            continue
        w = torch.tensor(
            [float(class_weights[dataset.class_to_id[b.category]]) for b in ann.bboxes]
        )
        sample_weights[i] = w.max() if aggregation == "max" else w.mean()
    return sample_weights


def make_weighted_sampler(
    dataset: BDD100KDetectionDataset,
    class_weights: Optional[torch.Tensor] = None,
    aggregation: str = "max",
) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` that oversamples rare-class images."""
    sample_weights = compute_sample_weights(dataset, class_weights, aggregation)
    return WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(dataset),
        replacement=True,
    )


def detection_collate_fn(
    batch: Sequence[Tuple[torch.Tensor, Target]],
) -> Tuple[List[torch.Tensor], List[Target]]:
    """Collate a batch of ``(image, target)`` pairs for detection.

    Object detection images have variable numbers of boxes, so images and
    targets are kept as lists (the standard torchvision detection convention)
    rather than stacked into a single tensor.
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloader(
    data_dir: str | Path,
    split: str = "train",
    batch_size: int = 4,
    img_size: int = 640,
    max_images: Optional[int] = None,
    use_weighted_sampler: bool = False,
    weight_scheme: str = "inverse",
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, torch.Tensor]:
    """Create a ``DataLoader`` for a BDD100K split and its class weights.

    Args:
        data_dir: Root data directory containing ``bdd100k/``.
        split: ``"train"``, ``"val"`` or ``"test"``.
        batch_size: Images per batch.
        img_size: Square resize side length.
        max_images: Optional cap on the number of images.
        use_weighted_sampler: If ``True``, oversample rare-class images via a
            ``WeightedRandomSampler`` (mutually exclusive with ``shuffle``).
        weight_scheme: Scheme passed to :func:`compute_class_weights`.
        shuffle: Shuffle when not using the weighted sampler.
        num_workers: ``DataLoader`` worker processes.
        pin_memory: Pin memory for faster host→GPU transfer.

    Returns:
        ``(dataloader, class_weights)`` where ``class_weights`` is a
        ``(NUM_CLASSES,)`` tensor suitable for a weighted loss.
    """
    dataset = BDD100KDetectionDataset(
        data_dir, split=split, img_size=img_size, max_images=max_images
    )
    class_weights = compute_class_weights(dataset, scheme=weight_scheme)

    sampler: Optional[WeightedRandomSampler] = None
    if use_weighted_sampler:
        sampler = make_weighted_sampler(dataset, class_weights)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=detection_collate_fn,
    )
    return loader, class_weights


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the data-loader demo."""
    parser = argparse.ArgumentParser(
        description="Demo the BDD100K PyTorch data loader (Dataset + weights).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--weight-scheme", type=str, default="inverse")
    parser.add_argument("--weighted-sampler", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Build the loader on a subset and print a summary of one batch."""
    args = _parse_args()

    print("=" * 64)
    print("BDD100K PyTorch DataLoader demo")
    print("=" * 64)

    loader, class_weights = build_dataloader(
        args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        img_size=args.img_size,
        max_images=args.max_images,
        use_weighted_sampler=args.weighted_sampler,
        weight_scheme=args.weight_scheme,
    )
    dataset: BDD100KDetectionDataset = loader.dataset  # type: ignore[assignment]

    print(f"\nSplit:            {args.split}")
    print(f"Dataset size:     {len(dataset)} images")
    print(f"Weighted sampler: {args.weighted_sampler}")

    counts = dataset.class_counts()
    print(f"\nPer-class object counts and '{args.weight_scheme}' weights:")
    print(f"  {'class':<14}{'count':>10}{'weight':>10}")
    for cid in range(NUM_CLASSES):
        name = dataset.id_to_class[cid]
        print(f"  {name:<14}{int(counts[cid]):>10}{class_weights[cid]:>10.3f}")

    images, targets = next(iter(loader))
    print(f"\nFirst batch: {len(images)} images")
    print(f"  image[0] tensor shape: {tuple(images[0].shape)} dtype={images[0].dtype}")
    print(f"  image[0] value range:  [{images[0].min():.3f}, {images[0].max():.3f}]")
    for i, tgt in enumerate(targets):
        print(
            f"  target[{i}]: boxes={tuple(tgt['boxes'].shape)} "
            f"labels={tgt['labels'].tolist()}"
        )

    print("\nData loader is working. ✅")


if __name__ == "__main__":
    main()
