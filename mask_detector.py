"""Face-mask detection, tracking, and annotated-output utilities."""

from __future__ import annotations

import os
import time
from typing import Any

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

COLOR_MASK = (80, 200, 80)
COLOR_NO_MASK = (60, 60, 235)
COLOR_UNKNOWN = (0, 180, 230)
COLOR_BG = (35, 35, 35)
COLOR_WHITE = (245, 245, 245)


def normalize_class_name(name: Any) -> str:
    """Normalize a model class name for reliable semantic matching."""
    return "_".join(str(name).strip().lower().replace("-", "_").split())


def classify_mask_label(class_name: str) -> str:
    """Map common dataset class names to Mask / No Mask / Mask Incorrect.

    Unknown class names intentionally remain unknown instead of being guessed.
    This prevents a custom model with unrelated labels from silently producing
    an incorrect Mask/No Mask UI.
    """
    n = normalize_class_name(class_name)

    no_mask = {
        "no_mask",
        "without_mask",
        "unmasked",
        "not_wearing_mask",
        "not_wearing_a_mask",
        "face_without_mask",
        "face_no_mask",
        "no_mask_face",
    }
    incorrect = {
        "mask_weared_incorrect",
        "mask_worn_incorrectly",
        "mask_worn_incorrect",
        "incorrect_mask",
        "mask_incorrect",
        "improper_mask",
        "improperly_worn_mask",
    }
    mask = {
        "mask",
        "with_mask",
        "wearing_mask",
        "wearing_a_mask",
        "face_mask",
        "masked",
        "proper_mask",
    }

    if n in no_mask or "without_mask" in n or n.startswith("no_mask"):
        return "No Mask"
    if n in incorrect or "incorrect" in n or "improper" in n:
        return "Mask Incorrect"
    if n in mask or n.endswith("_mask") and "no_" not in n and "without" not in n:
        return "Mask"
    return "Unknown"


def _label_color(class_name: str):
    semantic = classify_mask_label(class_name)
    if semantic == "Mask":
        return COLOR_MASK, semantic
    if semantic == "No Mask":
        return COLOR_NO_MASK, semantic
    if semantic == "Mask Incorrect":
        return COLOR_UNKNOWN, semantic
    return COLOR_UNKNOWN, str(class_name)


class MaskDetector:
    """Run YOLO face-mask detection on images, videos, or a webcam."""

    def __init__(
        self,
        model_path: str,
        conf: float = 0.4,
        iou: float = 0.5,
        title: str = "Face Mask Detection AI",
        imgsz: int = 640,
        device: str | None = None,
        max_det: int = 100,
        mask_class: str | None = None,
        no_mask_class: str | None = None,
        tracker: str = "bytetrack.yaml",
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at '{model_path}'. "
                "Run `python download_model.py` first or pass --model."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Run `python -m pip install -r requirements.txt`."
            ) from exc

        self.model_path = model_path
        self.model = YOLO(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.title = title
        self.imgsz = int(imgsz)
        self.device = device
        self.max_det = int(max_det)
        self.mask_class_override = normalize_class_name(mask_class) if mask_class else None
        self.no_mask_class_override = normalize_class_name(no_mask_class) if no_mask_class else None
        self.tracker = tracker
        self.track_labels: dict[int, str] = {}
        self.active_track_labels: dict[int, str] = {}
        self._validate_classes()

    @property
    def class_names(self) -> dict[int, str]:
        return {int(k): str(v) for k, v in self.model.names.items()}

    def _validate_classes(self) -> None:
        names = self.class_names
        if not names:
            raise ValueError("The loaded model has no class names.")

        if self.mask_class_override or self.no_mask_class_override:
            missing = []
            normalized = {normalize_class_name(v): k for k, v in names.items()}
            if self.mask_class_override and self.mask_class_override not in normalized:
                missing.append(f"mask class '{self.mask_class_override}'")
            if self.no_mask_class_override and self.no_mask_class_override not in normalized:
                missing.append(f"no-mask class '{self.no_mask_class_override}'")
            if missing:
                raise ValueError(
                    "Invalid class override(s): " + ", ".join(missing) +
                    f". Available classes: {list(names.values())}"
                )
            return

        semantic = {classify_mask_label(v) for v in names.values()}
        if "Mask" not in semantic or "No Mask" not in semantic:
            raise ValueError(
                "Could not safely identify both Mask and No Mask classes in the model. "
                f"Available classes: {list(names.values())}. "
                "Pass --mask-class and --no-mask-class with the exact class names."
            )

    def _semantic_label(self, class_name: str) -> str:
        n = normalize_class_name(class_name)
        if self.mask_class_override and n == self.mask_class_override:
            return "Mask"
        if self.no_mask_class_override and n == self.no_mask_class_override:
            return "No Mask"
        return classify_mask_label(class_name)

    def reset_tracking(self) -> None:
        """Reset application-side labels and Ultralytics tracker state."""
        self.track_labels.clear()
        self.active_track_labels.clear()
        predictor = getattr(self.model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor is not None else None
        if trackers:
            for tracker in trackers:
                reset = getattr(tracker, "reset", None)
                if callable(reset):
                    reset()
        # Recreating the predictor guarantees state isolation across files.
        if predictor is not None:
            try:
                self.model.predictor = None
            except Exception:
                pass

    def _inference_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        return kwargs

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _draw_top_bar(self, frame, counts: dict[str, int] | None = None):
        h, w = frame.shape[:2]
        bar_h = max(40, int(h * 0.07))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        font = cv2.FONT_HERSHEY_DUPLEX
        scale = bar_h / 55.0
        cv2.putText(frame, self.title, (16, int(bar_h * 0.68)), font,
                    scale * 0.75, COLOR_WHITE, 1, cv2.LINE_AA)

        counts = counts or {"Mask": 0, "No Mask": 0, "Mask Incorrect": 0, "Unknown": 0}
        text_mask = f"Mask: {counts['Mask']}"
        text_no_mask = f"No Mask: {counts['No Mask']}"
        font_scale = scale * 0.75
        (tw2, _), _ = cv2.getTextSize(text_no_mask, font, font_scale, 2)
        (tw1, _), _ = cv2.getTextSize(text_mask, font, font_scale, 2)
        pad = 24
        x2 = w - pad
        x1 = x2 - tw2
        x0 = x1 - pad - tw1

        cv2.putText(frame, text_mask, (max(8, x0), int(bar_h * 0.68)), font,
                    font_scale, COLOR_MASK, 2, cv2.LINE_AA)
        cv2.putText(frame, text_no_mask, (max(8, x1), int(bar_h * 0.68)), font,
                    font_scale, COLOR_NO_MASK, 2, cv2.LINE_AA)
        return bar_h

    @staticmethod
    def _counts_from_labels(labels: dict[int, str]) -> dict[str, int]:
        counts = {"Mask": 0, "No Mask": 0, "Mask Incorrect": 0, "Unknown": 0}
        for name in labels.values():
            counts[classify_mask_label(name)] += 1
        return counts

    def _draw_detection(self, frame, box_xyxy, track_id, class_name, conf, bar_h):
        x1, y1, x2, y2 = [int(v) for v in box_xyxy]
        color, label = _label_color(class_name)
        thickness = max(2, frame.shape[1] // 500)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        semantic = self._semantic_label(class_name)
        if semantic in {"Mask", "No Mask"}:
            label = semantic
        id_txt = f"#{track_id}  " if track_id is not None else ""
        tag = f"{id_txt}{label}  {conf * 100:.0f}%"
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = max(0.5, frame.shape[1] / 1400)
        (tw, th), _ = cv2.getTextSize(tag, font, scale, 1)

        tag_y1 = max(bar_h, y1 - th - 14)
        tag_y2 = tag_y1 + th + 14
        tag_x1 = max(0, min(x1, frame.shape[1] - tw - 16))
        tag_x2 = min(frame.shape[1], tag_x1 + tw + 16)
        cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
        cv2.putText(frame, tag, (tag_x1 + 8, tag_y2 - 8), font, scale,
                    COLOR_WHITE, 1, cv2.LINE_AA)

    def annotate(self, frame, result, tracking: bool):
        bar_h = self._draw_top_bar(frame, self._counts_from_labels(self.active_track_labels if tracking else {}))
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            if tracking:
                self.active_track_labels = {}
                self._draw_top_bar(frame, self._counts_from_labels({}))
            return frame

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if tracking and boxes.id is not None
            else [None] * len(cls)
        )

        current: dict[int, str] = {}
        image_labels: dict[int, str] = {}
        for index, (box, c, conf, tid) in enumerate(zip(xyxy, cls, confs, ids)):
            class_name = self.class_names[int(c)]
            semantic = self._semantic_label(class_name)
            if tid is not None:
                tid = int(tid)
                current[tid] = class_name
                self.track_labels[tid] = class_name
            else:
                image_labels[index] = class_name
            self._draw_detection(frame, box, tid, class_name, float(conf), bar_h)

        if tracking:
            self.active_track_labels = current
            counts = {
                "Mask": sum(self._semantic_label(v) == "Mask" for v in current.values()),
                "No Mask": sum(self._semantic_label(v) == "No Mask" for v in current.values()),
                "Mask Incorrect": sum(self._semantic_label(v) == "Mask Incorrect" for v in current.values()),
                "Unknown": sum(self._semantic_label(v) == "Unknown" for v in current.values()),
            }
        else:
            counts = {
                "Mask": sum(self._semantic_label(v) == "Mask" for v in image_labels.values()),
                "No Mask": sum(self._semantic_label(v) == "No Mask" for v in image_labels.values()),
                "Mask Incorrect": sum(self._semantic_label(v) == "Mask Incorrect" for v in image_labels.values()),
                "Unknown": sum(self._semantic_label(v) == "Unknown" for v in image_labels.values()),
            }
        bar_h = self._draw_top_bar(frame, counts)

        # Move tags away from the header after the final bar height is known.
        # Re-render boxes only in the uncommon case where the frame is small.
        _ = bar_h
        return frame

    # ------------------------------------------------------------------ #
    # Image
    # ------------------------------------------------------------------ #
    def process_image(self, in_path: str, out_dir: str) -> str:
        self.reset_tracking()
        frame = cv2.imread(in_path)
        if frame is None:
            raise ValueError(f"Could not read image: {in_path}")

        results = self.model.predict(source=frame, **self._inference_kwargs())
        annotated = self.annotate(frame, results[0], tracking=False)

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"{os.path.splitext(os.path.basename(in_path))[0]}_detected.jpg"
        )
        if not cv2.imwrite(out_path, annotated):
            raise IOError(f"Could not write output image: {out_path}")
        return out_path

    # ------------------------------------------------------------------ #
    # Video
    # ------------------------------------------------------------------ #
    def process_video(self, in_path: str, out_dir: str, show: bool = False) -> str:
        self.reset_tracking()
        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {in_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            cap.release()
            raise ValueError(f"Could not read video dimensions: {in_path}")

        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"{os.path.splitext(os.path.basename(in_path))[0]}_detected.mp4"
        )
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise IOError(f"Could not create output video: {out_path}")

        frame_idx = 0
        t0 = time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker,
                    **self._inference_kwargs(),
                )
                annotated = self.annotate(frame, results[0], tracking=True)
                writer.write(annotated)
                frame_idx += 1
                if show:
                    cv2.imshow("Face Mask Detection AI", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            writer.release()
            if show:
                cv2.destroyAllWindows()

        elapsed = time.time() - t0
        speed = frame_idx / max(elapsed, 1e-6)
        print(f"  processed {frame_idx} frames in {elapsed:.1f}s ({speed:.1f} fps) -> {out_path}")
        return out_path

    # ------------------------------------------------------------------ #
    # Webcam
    # ------------------------------------------------------------------ #
    def process_webcam(self, cam_index: int = 0, save_path: str | None = None):
        self.reset_tracking()
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {cam_index}. "
                "Check camera permissions or try another --cam-index."
            )

        writer = None
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            if not writer.isOpened():
                cap.release()
                raise IOError(f"Could not create webcam output: {save_path}")

        print("Live webcam started. Press 'q' in the video window to quit.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the webcam.")
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker,
                    **self._inference_kwargs(),
                )
                annotated = self.annotate(frame, results[0], tracking=True)
                if writer is not None:
                    writer.write(annotated)
                cv2.imshow("Face Mask Detection AI - Live Webcam", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
                print(f"Saved webcam recording -> {save_path}")
            cv2.destroyAllWindows()
