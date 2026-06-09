from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from src.data.parser import DETECTION_CLASSES

COCO_TO_BDD: Dict[str, str] = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "person": "person",
    "bicycle": "bike",
    "motorcycle": "motor",
    "traffic light": "traffic light",
}

# BDD classes with no COCO source class (cannot be predicted zero-shot).
BDD_UNMAPPED_FROM_COCO: List[str] = [
    c for c in DETECTION_CLASSES if c not in set(COCO_TO_BDD.values())
]  # -> ["rider", "traffic sign"]


def coco_name_to_bdd(coco_name: str) -> Optional[str]:
    """Map a COCO class name to its BDD100K equivalent, or ``None`` if absent."""
    return COCO_TO_BDD.get(coco_name.strip().lower())


RTDETR_ALIASES: Dict[str, str] = {
    "rtdetr-r18": "PekingU/rtdetr_r18vd",
    "rtdetr-r34": "PekingU/rtdetr_r34vd",
    "rtdetr-r50": "PekingU/rtdetr_r50vd",
    "rtdetr-r101": "PekingU/rtdetr_r101vd",
    "rtdetrv2-r18": "PekingU/rtdetr_v2_r18vd",
    "rtdetrv2-r34": "PekingU/rtdetr_v2_r34vd",
    "rtdetrv2-r50": "PekingU/rtdetr_v2_r50vd",
    "rtdetrv2-r101": "PekingU/rtdetr_v2_r101vd",
}
# Lightest, fastest real-time variant — a sensible default for a driving demo.
DEFAULT_RTDETR_MODEL = "rtdetr-r18"

TORCHVISION_ALIASES = {
    "retinanet": "retinanet_resnet50_fpn_v2",
    "fasterrcnn": "fasterrcnn_resnet50_fpn_v2",
    "ssdlite": "ssdlite320_mobilenet_v3_large",
}
DEFAULT_TORCHVISION_MODEL = "retinanet"

RTDETRV2_MODELS: Dict[str, Dict[str, object]] = {
    "rtdetrv2-s": {
        "config": "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
        "tag": "v0.2",
        "weight_file": "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        "ap": 48.1,
    },
    "rtdetrv2-m-r34": {
        "config": "configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml",
        "tag": "v0.1",
        "weight_file": "rtdetrv2_r34vd_120e_coco_ema.pth",
        "ap": 49.9,
    },
    "rtdetrv2-m": {
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml",
        "tag": "v0.1",
        "weight_file": "rtdetrv2_r50vd_m_7x_coco_ema.pth",
        "ap": 51.9,
    },
    "rtdetrv2-l": {
        "config": "configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",
        "tag": "v0.1",
        "weight_file": "rtdetrv2_r50vd_6x_coco_ema.pth",
        "ap": 53.4,
    },
    "rtdetrv2-x": {
        "config": "configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",
        "tag": "v0.1",
        "weight_file": "rtdetrv2_r101vd_6x_coco_from_paddle.pth",
        "ap": 54.3,
    },
}
# Lightest real-time variant (ResNet-18) — small download, fast CPU demo.
DEFAULT_RTDETRV2_GH_MODEL = "rtdetrv2-s"

DEFAULT_VENDOR_DIR = "third_party/RT-DETR/rtdetrv2_pytorch"
DEFAULT_WEIGHTS_DIR = "weights"

RELEASE_BASE_URL = "https://github.com/lyuwenyu/storage/releases/download"


def rtdetrv2_weight_url(model: str) -> str:
    """Return the GitHub-release download URL for an ``rtdetrv2-*`` model."""
    spec = RTDETRV2_MODELS[model]
    return f"{RELEASE_BASE_URL}/{spec['tag']}/{spec['weight_file']}"


VALID_BACKENDS = ("rtdetrv2-gh", "rtdetrv2-bdd", "yolo", "rtdetr", "torchvision")


@dataclass
class Detection:
    """A single detected object (absolute pixel xyxy box).

    Attributes:
        x1, y1, x2, y2: Box corners in pixels.
        score: Confidence in [0, 1].
        coco_label: Raw COCO class name emitted by the model.
        bdd_label: Mapped BDD100K class, or ``None`` if the COCO class has no
            BDD equivalent (e.g. COCO ``stop sign``, ``bench``, ...).
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    coco_label: str
    bdd_label: Optional[str] = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def in_bdd(self) -> bool:
        """True if this detection maps to one of the 10 BDD classes."""
        return self.bdd_label is not None

    def to_dict(self) -> dict:
        return {
            "bbox": [round(self.x1, 2), round(self.y1, 2),
                     round(self.x2, 2), round(self.y2, 2)],
            "score": round(float(self.score), 4),
            "coco_label": self.coco_label,
            "bdd_label": self.bdd_label,
        }


def filter_to_bdd(detections: Sequence[Detection]) -> List[Detection]:
    """Keep only detections whose COCO class maps to a BDD100K class."""
    return [d for d in detections if d.in_bdd]


_WIDE_BOX_REMAP: Dict[str, str] = {"motorcycle": "car", "bicycle": "car"}
_WIDE_BOX_RATIO = 1.2   # width / height threshold


def correct_domain_gap_labels(detections: List[Detection]) -> List[Detection]:
    """Reclassify wide motorcycle/bicycle boxes as cars.

    The COCO-pretrained RT-DETRv2 has a domain-gap issue on BDD100K: it fires
    the *motorcycle* decoder for cars in driving scenes, producing many
    landscape-shaped "motorcycle" predictions. Genuine motorcycles and bicycles
    in dashcam footage are portrait (height > width). Correcting landscape
    predictions to "car" significantly reduces false positives without any
    fine-tuning.
    """
    out: List[Detection] = []
    for d in detections:
        if d.height > 0 and d.width / d.height >= _WIDE_BOX_RATIO:
            new_coco = _WIDE_BOX_REMAP.get(d.coco_label)
            if new_coco is not None:
                d = Detection(
                    x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                    score=d.score,
                    coco_label=new_coco,
                    bdd_label=coco_name_to_bdd(new_coco),
                )
        out.append(d)
    return out


def detections_from_records(
    records: Sequence[dict], score_thresh: float = 0.0
) -> List[Detection]:
    """Build ``Detection`` objects from raw ``{coco_label, score, bbox}`` dicts.

    This is the pure-logic bridge used by the GitHub RT-DETRv2 backend (whose
    subprocess emits JSON) — kept separate so it can be unit-tested without any
    torch / network / subprocess. Each record needs ``coco_label`` (str),
    ``score`` (float) and ``bbox`` ([x1, y1, x2, y2]); the COCO→BDD mapping is
    applied here.
    """
    out: List[Detection] = []
    for r in records:
        score = float(r["score"])
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = (float(v) for v in r["bbox"])
        coco_label = str(r.get("coco_label", "")).lower()
        # Fine-tuned models write ``bdd_label`` directly; pretrained COCO models
        # write ``coco_label`` which we map here.
        bdd_label = r.get("bdd_label") or coco_name_to_bdd(coco_label)
        out.append(
            Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                score=score,
                coco_label=coco_label or (bdd_label or "unknown"),
                bdd_label=bdd_label,
            )
        )
    return out

def _apply_ssl_settings(ca_bundle: Optional[str] = None,
                        insecure: bool = False) -> None:
    """Make weight downloads work behind an SSL-intercepting corporate proxy.

    Preferred (secure): point every HTTP client at a CA bundle that includes the
    corporate root certificate (e.g. the host's
    ``/etc/ssl/certs/ca-certificates.crt`` mounted into the container).

    Last resort (opt-in): ``insecure=True`` disables certificate verification for
    ``urllib`` (torch.hub / torchvision), ``requests`` *and* ``httpx`` — the last
    of which is what modern ``huggingface_hub`` uses to fetch model weights.
    """
    if ca_bundle:
        ca_bundle = str(Path(ca_bundle).expanduser())
        for var in ("REQUESTS_CA_BUNDLE", "SSL_CERT_FILE", "CURL_CA_BUNDLE"):
            os.environ[var] = ca_bundle

    if insecure:
        print("[detector] WARNING: --insecure disables TLS verification "
              "(use only on a trusted corporate network).", file=sys.stderr)
        import ssl

        # urllib (torch.hub checkpoint download, torchvision weights)
        ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[attr-defined]

        # Drop any CA-bundle envs so nothing re-enables verification.
        for var in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
            os.environ.pop(var, None)

        try:  # pragma: no cover - network/runtime path
            import requests
            import urllib3

            urllib3.disable_warnings()
            _orig = requests.Session.merge_environment_settings

            def _no_verify(self, url, proxies, stream, verify, cert):
                settings = _orig(self, url, proxies, stream, verify, cert)
                settings["verify"] = False
                return settings

            requests.Session.merge_environment_settings = _no_verify  # type: ignore[assignment]
        except Exception:  # pragma: no cover
            pass

        try:  # pragma: no cover - network/runtime path
            import warnings

            import httpx

            warnings.filterwarnings("ignore")
            for _cls_name in ("Client", "AsyncClient"):
                _cls = getattr(httpx, _cls_name, None)
                if _cls is None:
                    continue
                _orig_init = _cls.__init__

                def _make_init(orig_init):
                    def _patched_init(self, *a, **k):
                        k["verify"] = False
                        return orig_init(self, *a, **k)
                    return _patched_init

                _cls.__init__ = _make_init(_orig_init)  # type: ignore[method-assign]
        except Exception:  # pragma: no cover
            pass


def _resolve_device(device: Optional[str]) -> str:
    """Resolve 'cuda'/'cpu'/None to an available device string."""
    import torch  # lazy

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"

class Detector:
    """Abstract base class for a pretrained, pure-PyTorch detector."""

    backend: str = "base"
    model_id: str = ""

    def __init__(self, score_thresh: float = 0.3, device: Optional[str] = None):
        self.score_thresh = score_thresh
        self.device = device or "cpu"

    def predict(self, image) -> List[Detection]:  # pragma: no cover - abstract
        """Run detection on a single PIL.Image and return Detections."""
        raise NotImplementedError

    def predict_path(self, image_path) -> List[Detection]:
        """Open an image file and run :meth:`predict`."""
        from PIL import Image

        with Image.open(image_path) as im:
            return self.predict(im.convert("RGB"))


class RTDetrDetector(Detector):
    """RT-DETR / RT-DETRv2 via HuggingFace ``transformers`` (pure PyTorch)."""

    backend = "rtdetr"

    def __init__(
        self,
        model: str = DEFAULT_RTDETR_MODEL,
        score_thresh: float = 0.3,
        device: Optional[str] = None,
        local_files_only: bool = False,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(score_thresh=score_thresh, device=device)
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.model_id = RTDETR_ALIASES.get(model, model)
        self.device = _resolve_device(device)

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_id, local_files_only=local_files_only, cache_dir=cache_dir
        )
        self.model = AutoModelForObjectDetection.from_pretrained(
            self.model_id, local_files_only=local_files_only, cache_dir=cache_dir
        )
        self.model.to(self.device).eval()
        self._torch = torch
        # COCO id -> name straight from the checkpoint config.
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def predict(self, image) -> List[Detection]:
        torch = self._torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # (height, width) target for box rescaling back to original pixels.
        target_sizes = torch.tensor([[image.height, image.width]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.score_thresh
        )[0]

        detections: List[Detection] = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            coco_name = self.id2label.get(int(label), str(int(label))).lower()
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            detections.append(
                Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    score=float(score),
                    coco_label=coco_name,
                    bdd_label=coco_name_to_bdd(coco_name),
                )
            )
        return detections


class TorchvisionDetector(Detector):
    """RetinaNet / Faster R-CNN / SSDLite from ``torchvision`` (pure PyTorch)."""

    backend = "torchvision"

    def __init__(
        self,
        model: str = DEFAULT_TORCHVISION_MODEL,
        score_thresh: float = 0.3,
        device: Optional[str] = None,
    ):
        super().__init__(score_thresh=score_thresh, device=device)
        import torch
        from torchvision.models import detection as tv_detection
        from torchvision.models import get_model_weights

        fn_name = TORCHVISION_ALIASES.get(model, model)
        if not hasattr(tv_detection, fn_name):
            raise ValueError(
                f"Unknown torchvision model '{model}'. "
                f"Choose from {sorted(TORCHVISION_ALIASES)} or a "
                f"torchvision.models.detection factory name."
            )
        build_fn = getattr(tv_detection, fn_name)

        self.model_id = fn_name
        self.device = _resolve_device(device)

        weights_enum = get_model_weights(fn_name).DEFAULT
        net = build_fn(weights=weights_enum)
        net.to(self.device).eval()
        self.model = net
        self._torch = torch
        self.categories = list(weights_enum.meta["categories"])

    def predict(self, image) -> List[Detection]:
        torch = self._torch
        from torchvision.transforms.functional import to_tensor

        img_t = to_tensor(image).to(self.device)
        with torch.no_grad():
            out = self.model([img_t])[0]

        detections: List[Detection] = []
        for box, label, score in zip(out["boxes"], out["labels"], out["scores"]):
            s = float(score)
            if s < self.score_thresh:
                continue
            idx = int(label)
            coco_name = (
                self.categories[idx] if 0 <= idx < len(self.categories) else str(idx)
            ).lower()
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            detections.append(
                Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    score=s,
                    coco_label=coco_name,
                    bdd_label=coco_name_to_bdd(coco_name),
                )
            )
        return detections


class RTDetrV2GithubDetector(Detector):

    backend = "rtdetrv2-gh"

    def __init__(
        self,
        model: str = DEFAULT_RTDETRV2_GH_MODEL,
        score_thresh: float = 0.3,
        device: Optional[str] = None,
        vendor_dir: Optional[str] = None,
        weights_dir: Optional[str] = None,
        image_size: int = 640,
    ):
        super().__init__(score_thresh=score_thresh, device=device)
        if model not in RTDETRV2_MODELS:
            raise ValueError(
                f"Unknown rtdetrv2 model '{model}'. "
                f"Choose from {sorted(RTDETRV2_MODELS)}."
            )
        self.model = model
        self.model_id = model
        self.image_size = image_size
        self.device = device or "cpu"  # no torch import here; subprocess resolves

        spec = RTDETRV2_MODELS[model]
        self.vendor_dir = Path(
            vendor_dir or os.environ.get("RTDETRV2_DIR") or DEFAULT_VENDOR_DIR
        )
        self.weights_dir = Path(
            weights_dir or os.environ.get("RTDETRV2_WEIGHTS_DIR") or DEFAULT_WEIGHTS_DIR
        )
        self.config_path = self.vendor_dir / str(spec["config"])
        self.weight_path = self.weights_dir / str(spec["weight_file"])

        # Fail early, with actionable guidance, if assets are missing.
        missing = []
        if not self.vendor_dir.is_dir():
            missing.append(f"source dir '{self.vendor_dir}'")
        elif not self.config_path.is_file():
            missing.append(f"config '{self.config_path}'")
        if not self.weight_path.is_file():
            missing.append(f"checkpoint '{self.weight_path}'")
        if missing:
            raise FileNotFoundError(
                "RT-DETRv2 (GitHub) assets missing: " + "; ".join(missing) + ".\n"
                f"Fetch them on a host that can reach github.com:\n"
                f"    scripts/setup_rtdetrv2.sh {model}\n"
                f"then mount third_party/ and weights/ into the container."
            )

        self._runner = Path(__file__).resolve().parent.parent / "scripts" / "rtdetrv2_infer.py"
        if not self._runner.is_file():
            raise FileNotFoundError(f"Inference runner not found: {self._runner}")

    def _extra_infer_args(self) -> List[str]:
        return []

    def predict_path(self, image_path) -> List[Detection]:
        """Run the official inference subprocess on an image file."""
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "dets.json"
            env = dict(os.environ)
            env["RTDETRV2_DIR"] = str(self.vendor_dir.resolve())
            cmd = [
                sys.executable, str(self._runner),
                "-c", str(self.config_path),
                "-r", str(self.weight_path),
                "-f", str(image_path),
                "-o", str(out_json),
                "-d", self.device,
                "--size", str(self.image_size),
            ] + self._extra_infer_args()
            proc = subprocess.run(env=env, args=cmd, capture_output=True, text=True)
            if proc.returncode != 0 or not out_json.exists():
                raise RuntimeError(
                    "RT-DETRv2 inference subprocess failed "
                    f"(exit {proc.returncode}).\n--- stderr ---\n{proc.stderr}"
                )
            with open(out_json, "r", encoding="utf-8") as f:
                payload = json.load(f)

        return detections_from_records(
            payload.get("detections", []), score_thresh=self.score_thresh
        )

    def predict(self, image) -> List[Detection]:
        """Run on an in-memory PIL image by staging it to a temp file."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
            tmp = tf.name
        try:
            image.convert("RGB").save(tmp)
            return self.predict_path(tmp)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def batch_predict_paths(
        self,
        image_paths: Sequence[str],
        batch_size: int = 8,
        device: Optional[str] = None,
    ) -> Dict[str, List[Detection]]:
        """Run batch inference on many images with one subprocess call.

        The model is loaded once; images are processed in mini-batches of
        ``batch_size`` on GPU (if available) or CPU. This is orders of magnitude
        faster than calling :meth:`predict_path` once per image.

        Args:
            image_paths: Absolute paths to the images to run inference on.
            batch_size: Number of images per forward pass (8 works well on GPU).
            device: ``"cuda"``, ``"cpu"`` or ``None`` / ``"auto"`` for automatic
                selection. Falls back to the detector's own device when ``None``.

        Returns:
            Mapping from each image path (str) to its list of :class:`Detection`
            objects (after score thresholding and COCO→BDD mapping).
        """
        import subprocess
        import tempfile

        effective_device = device or self.device or "auto"

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            list_json = td_path / "image_list.json"
            out_json = td_path / "batch_results.json"

            with open(str(list_json), "w", encoding="utf-8") as f:
                json.dump(list(image_paths), f)

            env = dict(os.environ)
            env["RTDETRV2_DIR"] = str(self.vendor_dir.resolve())
            cmd = [
                sys.executable, str(self._runner),
                "-c", str(self.config_path),
                "-r", str(self.weight_path),
                "--image-list", str(list_json),
                "-o", str(out_json),
                "-d", effective_device,
                "--size", str(self.image_size),
                "--batch-size", str(batch_size),
            ] + self._extra_infer_args()
            proc = subprocess.run(env=env, args=cmd, capture_output=True, text=True)

            if proc.stderr:
                for line in proc.stderr.splitlines():
                    print(line, file=sys.stderr, flush=True)

            if proc.returncode != 0 or not out_json.exists():
                raise RuntimeError(
                    "RT-DETRv2 batch inference subprocess failed "
                    f"(exit {proc.returncode}).\n--- stderr ---\n{proc.stderr}"
                )

            with open(str(out_json), "r", encoding="utf-8") as f:
                raw: Dict[str, list] = json.load(f)

        return {
            path: detections_from_records(records, score_thresh=self.score_thresh)
            for path, records in raw.items()
        }


# ---------------------------------------------------------------------------
# Fine-tuned RT-DETRv2 (BDD100K, 10 classes)
# ---------------------------------------------------------------------------

class FinetunedRTDetrV2Detector(RTDetrV2GithubDetector):
    """RT-DETRv2 fine-tuned on BDD100K (10 output classes).

    Uses the same subprocess inference mechanism as :class:`RTDetrV2GithubDetector`
    but passes the fine-tuned checkpoint via ``--finetuned-weights``.  Because
    the model was trained directly on BDD100K, class indices map to BDD labels
    via ``ID_TO_CLASS`` — no COCO→BDD remapping is needed, and
    :func:`correct_domain_gap_labels` should *not* be applied to its output.

    Args:
        finetuned_weights: Path to the checkpoint produced by ``train.py``
            (``output/rtdetrv2_bdd100k/best.pth``).
        base_model: RT-DETRv2 variant whose config will be used as the
            architecture template (must match the config used during training).
        score_thresh: Minimum confidence score to keep a detection.
    """

    backend = "rtdetrv2-bdd"

    def __init__(
        self,
        finetuned_weights: str,
        base_model: str = DEFAULT_RTDETRV2_GH_MODEL,
        score_thresh: float = 0.3,
        device: Optional[str] = None,
        vendor_dir: Optional[str] = None,
        weights_dir: Optional[str] = None,
        image_size: int = 640,
    ):
        super().__init__(
            model=base_model,
            score_thresh=score_thresh,
            device=device,
            vendor_dir=vendor_dir,
            weights_dir=weights_dir,
            image_size=image_size,
        )
        self._finetuned_path = Path(finetuned_weights)
        if not self._finetuned_path.is_file():
            raise FileNotFoundError(
                f"Fine-tuned checkpoint not found: {self._finetuned_path}\n"
                "Run 'docker run ... bdd-pipeline train' first to produce it."
            )
        self.model_id = f"{base_model}-bdd ({self._finetuned_path.name})"

    def _extra_infer_args(self) -> List[str]:
        return ["--finetuned-weights", str(self._finetuned_path)]

def build_detector(
    backend: str = "rtdetrv2-gh",
    model: Optional[str] = None,
    score_thresh: float = 0.3,
    device: Optional[str] = None,
    ca_bundle: Optional[str] = None,
    insecure: bool = False,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
    vendor_dir: Optional[str] = None,
    weights_dir: Optional[str] = None,
    finetuned_weights: Optional[str] = None,
    yolo_weights: Optional[str] = None,
) -> Detector:
    """Build a pretrained detector.

    Args:
        backend: One of ``"rtdetrv2-gh"`` (COCO-pretrained, default),
            ``"rtdetrv2-bdd"`` (fine-tuned on BDD100K — pass
            ``finetuned_weights``), ``"yolo"`` (ultralytics YOLO — pass
            ``yolo_weights``), ``"rtdetr"`` (HuggingFace, blocked here) or
            ``"torchvision"``.
        model: Backend-specific model name / alias.
        score_thresh: Confidence threshold for kept detections.
        device: ``"cuda"`` / ``"cpu"`` / ``None`` (auto).
        finetuned_weights: Path to the BDD100K fine-tuned checkpoint (required
            for ``backend="rtdetrv2-bdd"``).
        yolo_weights: Path to a ``.pt`` file for the YOLO backend.
        ca_bundle: CA bundle path for SSL-intercepting proxies.
        insecure: Disable TLS verification (opt-in last resort).
        local_files_only: Use only cached weights (offline mode).
        cache_dir: HuggingFace cache directory override.

    Raises:
        ValueError: Unrecognised backend name or missing required weight path.
    """
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose one of {VALID_BACKENDS}."
        )

    if backend == "rtdetrv2-bdd":
        if not finetuned_weights:
            raise ValueError(
                "finetuned_weights is required for the 'rtdetrv2-bdd' backend. "
                "Pass the path to output/rtdetrv2_bdd100k/best.pth."
            )
        return FinetunedRTDetrV2Detector(
            finetuned_weights=finetuned_weights,
            base_model=model or DEFAULT_RTDETRV2_GH_MODEL,
            score_thresh=score_thresh,
            device=device,
            vendor_dir=vendor_dir,
            weights_dir=weights_dir,
        )

    if backend == "yolo":
        if not yolo_weights:
            raise ValueError(
                "yolo_weights is required for the 'yolo' backend. "
                "Pass the path to your .pt model file."
            )
        from src.model.yolo_detector import YOLODetector
        return YOLODetector(weights=yolo_weights, score_thresh=score_thresh, device=device)

    # The GitHub RT-DETRv2 backend needs no in-process weight download.
    if backend == "rtdetrv2-gh":
        return RTDetrV2GithubDetector(
            model=model or DEFAULT_RTDETRV2_GH_MODEL,
            score_thresh=score_thresh,
            device=device,
            vendor_dir=vendor_dir,
            weights_dir=weights_dir,
        )

    # Apply SSL settings before any download is attempted.
    _apply_ssl_settings(ca_bundle=ca_bundle, insecure=insecure)

    if backend == "rtdetr":
        return RTDetrDetector(
            model=model or DEFAULT_RTDETR_MODEL,
            score_thresh=score_thresh,
            device=device,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
        )
    return TorchvisionDetector(
        model=model or DEFAULT_TORCHVISION_MODEL,
        score_thresh=score_thresh,
        device=device,
    )


_BDD_COLORS = {
    "car": (0, 200, 0),
    "truck": (0, 128, 255),
    "bus": (255, 128, 0),
    "train": (160, 0, 255),
    "person": (255, 0, 0),
    "rider": (255, 0, 200),
    "bike": (0, 255, 200),
    "motor": (255, 200, 0),
    "traffic light": (255, 255, 0),
    "traffic sign": (0, 255, 255),
}
_OTHER_COLOR = (150, 150, 150)


def draw_detections(image, detections: Sequence[Detection], bdd_only: bool = False):
    """Return a copy of ``image`` (PIL) with boxes + labels drawn."""
    from PIL import ImageDraw, ImageFont

    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None

    for d in detections:
        if bdd_only and not d.in_bdd:
            continue
        label = d.bdd_label or d.coco_label
        color = _BDD_COLORS.get(d.bdd_label or "", _OTHER_COLOR)
        draw.rectangle([d.x1, d.y1, d.x2, d.y2], outline=color, width=2)
        text = f"{label} {d.score:.2f}"
        ty = max(0, d.y1 - 11)
        draw.text((d.x1 + 2, ty), text, fill=color, font=font)
    return out


def _default_val_image(data_dir: str) -> Optional[Path]:
    """Return the first BDD100K val image found under ``data_dir``."""
    candidates = [
        Path(data_dir) / "bdd100k" / "images" / "100k" / "val",
        Path(data_dir) / "images" / "100k" / "val",
    ]
    for d in candidates:
        if d.is_dir():
            for p in sorted(d.glob("*.jpg")):
                return p
    return None


def _summarize(detections: Sequence[Detection]) -> Dict[str, int]:
    """Count detections per BDD class (ignores non-BDD COCO classes)."""
    counts: Dict[str, int] = {}
    for d in detections:
        if d.bdd_label:
            counts[d.bdd_label] = counts.get(d.bdd_label, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.detector",
        description="Pure-PyTorch pretrained detection for BDD100K — no "
                    "Ultralytics. Default backend 'rtdetrv2-gh' uses the OFFICIAL "
                    "RT-DETRv2 source + weights from GitHub).",
    )
    p.add_argument("--image", type=str, default=None,
                   help="Path to an image. Default: first val image under --data-dir.")
    p.add_argument("--data-dir", type=str, default="data",
                   help="Dataset root (used only to pick a default image).")
    p.add_argument("--backend", choices=VALID_BACKENDS, default="rtdetrv2-gh",
                   help="Detector backend (default: rtdetrv2-gh = official "
                        "GitHub RT-DETRv2).")
    p.add_argument("--model", type=str, default=None,
                   help="Model alias/name. rtdetrv2-gh: rtdetrv2-s|-m-r34|-m|-l|-x; "
                        "rtdetr (HF): rtdetr-r18|-r50; "
                        "torchvision: retinanet|fasterrcnn|ssdlite.")
    p.add_argument("--score", type=float, default=0.3,
                   help="Confidence threshold (default: 0.3).")
    p.add_argument("--device", type=str, default=None,
                   help="cuda | cpu (default: auto-detect).")
    p.add_argument("--bdd-only", action="store_true",
                   help="Keep only detections that map to the 10 BDD classes.")
    p.add_argument("--output", type=str, default="output",
                   help="Directory for the annotated image + JSON (default: output).")
    p.add_argument("--no-save", action="store_true",
                   help="Do not write annotated image / JSON.")
    # GitHub RT-DETRv2 asset locations (populated by scripts/setup_rtdetrv2.sh).
    p.add_argument("--vendor-dir", type=str, default=None,
                   help="Path to the vendored rtdetrv2_pytorch dir "
                        f"(default: {DEFAULT_VENDOR_DIR} or $RTDETRV2_DIR).")
    p.add_argument("--weights-dir", type=str, default=None,
                   help="Dir holding the .pth checkpoint "
                        f"(default: {DEFAULT_WEIGHTS_DIR} or $RTDETRV2_WEIGHTS_DIR).")
    # Proxy / SSL handling (rtdetr/torchvision backends only).
    p.add_argument("--ca-bundle", type=str, default=None,
                   help="CA bundle trusting the corporate proxy (secure fix).")
    p.add_argument("--insecure", action="store_true",
                   help="Disable TLS verification for weight download (last resort).")
    p.add_argument("--local-files-only", action="store_true",
                   help="Use only cached weights (offline).")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="HuggingFace cache directory override.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    image_path = Path(args.image) if args.image else _default_val_image(args.data_dir)
    if image_path is None or not Path(image_path).exists():
        print(f"[detector] No image found (looked for --image and under "
              f"'{args.data_dir}/.../images/100k/val'). Provide --image.",
              file=sys.stderr)
        return 2

    print(f"[detector] backend={args.backend} model={args.model or 'default'} "
          f"image={image_path}")
    detector = build_detector(
        backend=args.backend,
        model=args.model,
        score_thresh=args.score,
        device=args.device,
        ca_bundle=args.ca_bundle,
        insecure=args.insecure,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
        vendor_dir=args.vendor_dir,
        weights_dir=args.weights_dir,
    )
    print(f"[detector] loaded '{detector.model_id}' on {detector.device}")

    detections = detector.predict_path(image_path)
    if args.bdd_only:
        detections = filter_to_bdd(detections)

    counts = _summarize(detections)
    print(f"[detector] {len(detections)} detections "
          f"({sum(counts.values())} in BDD classes)")
    for cls, n in counts.items():
        print(f"    {cls:14s} {n}")
    unmapped = sorted({d.coco_label for d in detections if not d.in_bdd})
    if unmapped and not args.bdd_only:
        print(f"    [non-BDD COCO classes seen: {', '.join(unmapped)}]")
    print(f"    [note: COCO cannot produce {BDD_UNMAPPED_FROM_COCO} zero-shot]")

    if not args.no_save:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem

        json_path = out_dir / f"{stem}_detections.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image": str(image_path),
                    "backend": args.backend,
                    "model": detector.model_id,
                    "score_thresh": args.score,
                    "counts": counts,
                    "detections": [d.to_dict() for d in detections],
                },
                f,
                indent=2,
            )

        from PIL import Image

        with Image.open(image_path) as im:
            annotated = draw_detections(im, detections, bdd_only=args.bdd_only)
        img_out = out_dir / f"{stem}_detections.jpg"
        annotated.save(img_out)
        print(f"[detector] wrote {json_path} and {img_out}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
