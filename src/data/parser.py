import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# The 10 object detection classes in BDD100K
DETECTION_CLASSES = [
    "car",
    "truck",
    "bus",
    "train",
    "person",
    "rider",
    "bike",
    "motor",
    "traffic light",
    "traffic sign",
]


@dataclass
class BoundingBox:
    """Represents a single bounding box annotation.

    Attributes:
        x1: Left x-coordinate.
        y1: Top y-coordinate.
        x2: Right x-coordinate.
        y2: Bottom y-coordinate.
        category: Object class label.
        occluded: Whether the object is occluded.
        truncated: Whether the object is truncated.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    category: str
    occluded: bool = False
    truncated: bool = False

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Area of the bounding box in pixels."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width / height)."""
        if self.height == 0:
            return 0.0
        return self.width / self.height


@dataclass
class ImageAnnotation:
    """Represents all detection annotations for a single image.

    Attributes:
        filename: Name of the image file.
        width: Image width in pixels.
        height: Image height in pixels.
        bboxes: List of bounding box annotations.
        attributes: Scene-level attributes (weather, time of day, scene).
    """

    filename: str
    width: int = 1280
    height: int = 720
    bboxes: List[BoundingBox] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)


def parse_single_label(label_path: Path) -> ImageAnnotation:
    """Parse a single per-image BDD100K JSON label file.

    Handles the per-image format with structure:
        { "name": ..., "frames": [{ "objects": [...] }] }

    Only bounding boxes belonging to the 10 detection classes are retained.

    Args:
        label_path: Path to a single JSON label file.

    Returns:
        ImageAnnotation for that image.
    """
    with open(label_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    img_ann = ImageAnnotation(
        filename=raw_data.get("name", label_path.stem),
        attributes=raw_data.get("attributes", {}),
    )

    # Per-image format: frames[].objects[]
    frames = raw_data.get("frames", [])
    for frame in frames:
        objects = frame.get("objects", [])
        for obj in objects:
            category = obj.get("category", "")
            if category not in DETECTION_CLASSES:
                continue

            box2d = obj.get("box2d")
            if box2d is None:
                continue

            attrs = obj.get("attributes", {})
            bbox = BoundingBox(
                x1=float(box2d.get("x1", 0)),
                y1=float(box2d.get("y1", 0)),
                x2=float(box2d.get("x2", 0)),
                y2=float(box2d.get("y2", 0)),
                category=category,
                occluded=attrs.get("occluded", False),
                truncated=attrs.get("truncated", False),
            )
            img_ann.bboxes.append(bbox)

    # Fallback: flat "labels" array format (older BDD100K style)
    if not frames:
        labels = raw_data.get("labels", [])
        if labels is None:
            labels = []
        for label in labels:
            category = label.get("category", "")
            if category not in DETECTION_CLASSES:
                continue

            box2d = label.get("box2d")
            if box2d is None:
                continue

            bbox = BoundingBox(
                x1=float(box2d.get("x1", 0)),
                y1=float(box2d.get("y1", 0)),
                x2=float(box2d.get("x2", 0)),
                y2=float(box2d.get("y2", 0)),
                category=category,
                occluded=label.get("attributes", {}).get("occluded", False),
                truncated=label.get("attributes", {}).get("truncated", False),
            )
            img_ann.bboxes.append(bbox)

    return img_ann


def parse_bdd_labels(label_path: str) -> List[ImageAnnotation]:
    """Parse BDD100K label data from a single combined JSON file.

    Supports the combined-file format where one JSON contains an array
    of image entries each with a "labels" list.

    Args:
        label_path: Path to the combined BDD100K JSON label file.

    Returns:
        List of ImageAnnotation objects, one per image.

    Raises:
        FileNotFoundError: If the label file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # If it's a single object (per-image file), wrap it
    if isinstance(raw_data, dict):
        return [parse_single_label(label_path)]

    annotations = []
    for entry in raw_data:
        img_ann = ImageAnnotation(
            filename=entry.get("name", ""),
            attributes=entry.get("attributes", {}),
        )

        labels = entry.get("labels", [])
        if labels is None:
            labels = []

        for label in labels:
            category = label.get("category", "")
            if category not in DETECTION_CLASSES:
                continue

            box2d = label.get("box2d")
            if box2d is None:
                continue

            bbox = BoundingBox(
                x1=float(box2d.get("x1", 0)),
                y1=float(box2d.get("y1", 0)),
                x2=float(box2d.get("x2", 0)),
                y2=float(box2d.get("y2", 0)),
                category=category,
                occluded=label.get("attributes", {}).get("occluded", False),
                truncated=label.get("attributes", {}).get("truncated", False),
            )
            img_ann.bboxes.append(bbox)

        annotations.append(img_ann)

    return annotations


def load_split(
    data_dir: str, split: str = "train"
) -> Optional[List[ImageAnnotation]]:
    """Load annotations for a given split (train or val).

    Supports both:
      - Single combined JSON file (bdd100k_labels_images_train.json)
      - Directory of per-image JSON files (labels/100k/train/*.json)

    Args:
        data_dir: Root data directory containing bdd100k/.
        split: Dataset split: 'train', 'val', or 'test'.

    Returns:
        List of ImageAnnotation objects, or None if not found.
    """
    data_dir = Path(data_dir)

    # Check for per-image JSON directory (labels/100k/<split>/)
    per_image_dirs = [
        data_dir / "bdd100k" / "labels" / "100k" / split,
        data_dir / "labels" / "100k" / split,
    ]
    for dir_path in per_image_dirs:
        if dir_path.is_dir():
            json_files = sorted(dir_path.glob("*.json"))
            if json_files:
                print(f"Loading {split} labels from directory: {dir_path} ({len(json_files)} files)")
                annotations = []
                for jf in json_files:
                    annotations.append(parse_single_label(jf))
                return annotations

    # Check for single combined JSON file
    possible_paths = [
        data_dir / "bdd100k" / "labels" / f"bdd100k_labels_images_{split}.json",
        data_dir / "labels" / f"bdd100k_labels_images_{split}.json",
        data_dir / f"bdd100k_labels_images_{split}.json",
    ]

    for path in possible_paths:
        if path.exists():
            print(f"Loading {split} labels from: {path}")
            return parse_bdd_labels(str(path))

    print(f"Warning: Could not find {split} label file in {data_dir}")
    return None
