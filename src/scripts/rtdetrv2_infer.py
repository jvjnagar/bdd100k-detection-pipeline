import argparse
import json
import os
import sys
from pathlib import Path

# BDD100K 10-class label list — matches CLASS_TO_ID order in src/model/class_map.py.
# Used when running inference with a fine-tuned checkpoint (num_classes=10).
BDD_NAMES = [
    "car", "truck", "bus", "train", "person",
    "rider", "bike", "motor", "traffic light", "traffic sign",
]

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def _resolve_device(device: str) -> str:
    import torch
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_model(
    vendor_dir: Path,
    config_path: Path,
    weight_path: Path,
    device: str,
    num_classes: int = 80,
    finetuned: bool = False,
):
    """Load an RT-DETRv2 model.

    Args:
        num_classes: 80 for COCO-pretrained, 10 for BDD100K fine-tuned.
        finetuned: When True, use ``checkpoint['model']`` (from train.py) and
            build config with the matching num_classes.
    """
    vendor_src = str(vendor_dir.resolve())
    if vendor_src not in sys.path:
        sys.path.insert(0, vendor_src)

    import torch
    from src.core import YAMLConfig

    if finetuned:
        cfg = YAMLConfig(
            str(config_path),
            num_classes=num_classes,
            remap_mscoco_category=False,
            pretrained=False,
        )
        if "PResNet" in cfg.yaml_cfg:
            cfg.yaml_cfg["PResNet"]["pretrained"] = False
        if not hasattr(cfg, "model"):
            raise RuntimeError("YAMLConfig did not build a model attribute")
        # Weights are loaded by the caller (main) right after this returns.
    else:
        cfg = YAMLConfig(str(config_path), resume=str(weight_path))
        if not hasattr(cfg, "model"):
            raise RuntimeError("YAMLConfig did not build a model attribute")
        checkpoint = torch.load(str(weight_path), map_location="cpu")
        state = checkpoint.get("ema", {}).get("module", checkpoint.get("model", checkpoint))
        cfg.model.load_state_dict(state, strict=False)

    model = cfg.model.to(device).eval()
    postprocessor = getattr(cfg, "postprocessor", None)
    return model, postprocessor


def _decode_outputs(outputs, postprocessor, orig_sizes, device):
    """Decode model outputs into (labels, scores, boxes) lists.

    ``orig_sizes`` must be a tensor of shape [batch, 2] in [width, height] order
    to match the convention of RTDETRPostProcessor (taken from coco_dataset.py).
    """
    import torch

    if postprocessor is not None:
        results = postprocessor(outputs, orig_sizes)
        labels = results[0]["labels"].cpu().tolist()
        boxes  = results[0]["boxes"].cpu().tolist()
        scores = results[0]["scores"].cpu().tolist()
        return labels, scores, boxes

    # Fallback: decode from raw pred_logits / pred_boxes
    logits = outputs["pred_logits"][0]         # (num_queries, num_classes+1)
    raw_boxes = outputs["pred_boxes"][0]       # (num_queries, 4) normalised cxcywh

    probs = logits.softmax(-1)[..., :-1]       # drop background
    scores_t, labels_t = probs.max(-1)

    # orig_sizes[0] = [w, h] — same order as stored by coco_dataset.py
    orig_w, orig_h = orig_sizes[0].tolist()
    cx, cy, w, h = raw_boxes.unbind(-1)
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    boxes_px = torch.stack([x1, y1, x2, y2], dim=-1)

    return labels_t.cpu().tolist(), scores_t.cpu().tolist(), boxes_px.cpu().tolist()


def _dets_from_raw(labels, scores, boxes):
    """Format detections for a COCO-pretrained model (80 classes)."""
    return [
        {
            "coco_label": COCO_NAMES[int(l)] if 0 <= int(l) < len(COCO_NAMES) else str(l),
            "score": round(float(s), 4),
            "bbox": [round(float(v), 2) for v in b],
        }
        for l, s, b in zip(labels, scores, boxes)
    ]


def _dets_from_raw_bdd(labels, scores, boxes):
    """Format detections for a BDD100K fine-tuned model (10 classes).

    Outputs ``bdd_label`` directly so ``detections_from_records`` can skip the
    COCO->BDD mapping and use the label as-is.
    """
    return [
        {
            "bdd_label": BDD_NAMES[int(l)] if 0 <= int(l) < len(BDD_NAMES) else "unknown",
            "score": round(float(s), 4),
            "bbox": [round(float(v), 2) for v in b],
        }
        for l, s, b in zip(labels, scores, boxes)
    ]


def _apply_nms(dets: list, iou_threshold: float = 0.5) -> list:
    if len(dets) <= 1:
        return dets
    import torch
    import torchvision

    boxes = torch.tensor(
        [[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]] for d in dets],
        dtype=torch.float32,
    )
    scores = torch.tensor([d["score"] for d in dets], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes, scores, iou_threshold)
    return [dets[i] for i in keep.tolist()]


def _infer_single(
    model, postprocessor, image_path: Path, device: str, size: int,
    use_bdd: bool = False,
) -> list:
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F

    with Image.open(str(image_path)) as im:
        orig_w, orig_h = im.size
        img_t = F.to_tensor(im.convert("RGB").resize((size, size))).unsqueeze(0).to(device)

    # [w, h] order — matches RTDETRPostProcessor convention (coco_dataset.py)
    orig_sizes = torch.tensor([[orig_w, orig_h]], device=device)
    with torch.no_grad():
        outputs = model(img_t)

    labels, scores, boxes = _decode_outputs(outputs, postprocessor, orig_sizes, device)
    raw_fn = _dets_from_raw_bdd if use_bdd else _dets_from_raw
    return _apply_nms(raw_fn(labels, scores, boxes))


def _infer_batch(
    model, postprocessor, image_paths: list, device: str, size: int,
    batch_size: int, use_bdd: bool = False,
) -> dict:
    """Run inference on all images, model loaded once. Returns {path: [dets]}."""
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F

    results = {}
    total = len(image_paths)

    for start in range(0, total, batch_size):
        chunk = image_paths[start:start + batch_size]
        batch_tensors = []
        orig_sizes = []
        valid_paths = []

        for p in chunk:
            try:
                with Image.open(p) as im:
                    orig_w, orig_h = im.size
                    t = F.to_tensor(im.convert("RGB").resize((size, size)))
                batch_tensors.append(t)
                # [w, h] order — matches RTDETRPostProcessor convention
                orig_sizes.append([orig_w, orig_h])
                valid_paths.append(p)
            except Exception as exc:
                print(f"[rtdetrv2_infer] skipping {p}: {exc}", file=sys.stderr)
                results[p] = []

        if not batch_tensors:
            continue

        imgs = torch.stack(batch_tensors).to(device)
        sizes_t = torch.tensor(orig_sizes, device=device)

        with torch.no_grad():
            outputs_batch = model(imgs)

        for i, path in enumerate(valid_paths):
            # Slice per-image outputs from the batch
            single_out = {
                k: v[i:i+1] if isinstance(v, torch.Tensor) and v.dim() > 0 else v
                for k, v in outputs_batch.items()
            }
            labels, scores, boxes = _decode_outputs(
                single_out, postprocessor, sizes_t[i:i+1], device
            )
            raw_fn = _dets_from_raw_bdd if use_bdd else _dets_from_raw
            results[path] = _apply_nms(raw_fn(labels, scores, boxes))

        done = min(start + batch_size, total)
        print(f"[rtdetrv2_infer] {done}/{total} images processed on {device}",
              file=sys.stderr, flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="RT-DETRv2 inference (single or batch)")
    parser.add_argument("-c", "--config",  required=True)
    parser.add_argument("-r", "--resume",  required=True)
    parser.add_argument("-o", "--output",  required=True)
    parser.add_argument("-d", "--device",  default="auto",
                        help="cuda | cpu | auto (default: auto)")
    parser.add_argument("--size",       type=int, default=640)
    # Single-image mode
    parser.add_argument("-f", "--file", default=None, help="Single input image path")
    # Batch mode
    parser.add_argument("--image-list", default=None,
                        help="JSON file containing list of image paths")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Mini-batch size for GPU inference (batch mode only)")
    parser.add_argument(
        "--finetuned-weights", default=None,
        help=(
            "Path to a BDD100K fine-tuned checkpoint (output of train.py). "
            "When given, the model is built with num_classes=10 and class indices "
            "are mapped to BDD100K names directly instead of COCO names."
        ),
    )
    args = parser.parse_args()

    if args.file is None and args.image_list is None:
        print("[rtdetrv2_infer] ERROR: provide -f (single) or --image-list (batch)",
              file=sys.stderr)
        sys.exit(1)

    vendor_dir  = Path(os.environ.get("RTDETRV2_DIR",
                                      "third_party/RT-DETR/rtdetrv2_pytorch"))
    config_path = Path(args.config)
    weight_path = Path(args.resume)
    output_path = Path(args.output)
    finetuned_path = Path(args.finetuned_weights) if args.finetuned_weights else None

    for p, label in [(config_path, "config"), (weight_path, "weights")]:
        if not p.exists():
            print(f"[rtdetrv2_infer] ERROR: {label} not found: {p}", file=sys.stderr)
            sys.exit(1)

    if finetuned_path is not None and not finetuned_path.exists():
        print(f"[rtdetrv2_infer] ERROR: finetuned weights not found: {finetuned_path}",
              file=sys.stderr)
        sys.exit(1)

    use_bdd = finetuned_path is not None
    device = _resolve_device(args.device)
    print(f"[rtdetrv2_infer] loading model on {device} ...", file=sys.stderr, flush=True)

    if use_bdd:
        model, postprocessor = _load_model(
            vendor_dir, config_path, weight_path, device,
            num_classes=len(BDD_NAMES), finetuned=True,
        )
        import torch
        ft_ckpt = torch.load(str(finetuned_path), map_location=device)
        state = (
            ft_ckpt.get("ema", {}).get("module")
            or ft_ckpt.get("model")
            or ft_ckpt
        )
        model.load_state_dict(state, strict=False)
        print(f"[rtdetrv2_infer] fine-tuned weights loaded from {finetuned_path}",
              file=sys.stderr, flush=True)
    else:
        model, postprocessor = _load_model(vendor_dir, config_path, weight_path, device)

    print("[rtdetrv2_infer] model ready", file=sys.stderr, flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.file:
        image_path = Path(args.file)
        if not image_path.exists():
            print(f"[rtdetrv2_infer] ERROR: image not found: {image_path}",
                  file=sys.stderr)
            sys.exit(1)
        dets = _infer_single(model, postprocessor, image_path, device, args.size,
                              use_bdd=use_bdd)
        with open(str(output_path), "w", encoding="utf-8") as f:
            json.dump({"detections": dets}, f)
    else:
        with open(args.image_list, "r", encoding="utf-8") as f:
            image_paths = json.load(f)
        results = _infer_batch(
            model, postprocessor, image_paths, device, args.size, args.batch_size,
            use_bdd=use_bdd,
        )
        with open(str(output_path), "w", encoding="utf-8") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
