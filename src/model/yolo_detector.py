from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.model.detector import Detection, Detector, coco_name_to_bdd


_BDD_CLASS_SET = {
    "car", "truck", "bus", "train", "person",
    "rider", "bike", "motor", "traffic light", "traffic sign",
}


class YOLODetector(Detector):
    """Ultralytics YOLO wrapper that exposes the same interface as the RT-DETRv2
    backends so it can be dropped into the evaluation comparison loop.

    Args:
        weights: Path to a YOLO ``.pt`` checkpoint.
        score_thresh: Minimum confidence to keep a detection.
        device: ``"cuda"`` / ``"cpu"`` / ``None`` (auto-detect).
    """

    backend = "yolo"

    def __init__(
        self,
        weights: str,
        score_thresh: float = 0.3,
        device: Optional[str] = None,
    ):
        super().__init__(score_thresh=score_thresh, device=device)
        self._weights = str(weights)
        self._yolo = None
        self._is_bdd_model: Optional[bool] = None
        self.model_id = Path(weights).stem

    def _get_yolo(self):
        if self._yolo is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ImportError(
                    "ultralytics is not installed — run: pip install ultralytics"
                ) from exc
            self._yolo = YOLO(self._weights)
            model_classes = {n.lower() for n in self._yolo.names.values()}
            self._is_bdd_model = (
                model_classes <= _BDD_CLASS_SET or _BDD_CLASS_SET <= model_classes
            )
        return self._yolo

    def _parse_result(self, result) -> List[Detection]:
        yolo = self._get_yolo()
        dets: List[Detection] = []
        for box in result.boxes:
            score = float(box.conf.item())
            if score < self.score_thresh:
                continue
            cls_name = yolo.names[int(box.cls.item())].lower()
            bdd_label = (
                cls_name if (self._is_bdd_model and cls_name in _BDD_CLASS_SET)
                else coco_name_to_bdd(cls_name)
            )
            xyxy = box.xyxy[0].tolist()
            dets.append(Detection(
                x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                score=score,
                coco_label=cls_name,
                bdd_label=bdd_label,
            ))
        return dets

    def predict_path(self, image_path) -> List[Detection]:
        yolo = self._get_yolo()
        return self._parse_result(yolo(str(image_path), conf=self.score_thresh, verbose=False)[0])

    def predict(self, image) -> List[Detection]:
        yolo = self._get_yolo()
        return self._parse_result(yolo(image, conf=self.score_thresh, verbose=False)[0])

    def batch_predict_paths(
        self,
        image_paths: Sequence[str],
        batch_size: int = 8,
        device: Optional[str] = None,
    ) -> Dict[str, List[Detection]]:
        yolo = self._get_yolo()
        return {
            p: self._parse_result(yolo(p, conf=self.score_thresh, verbose=False)[0])
            for p in image_paths
        }
