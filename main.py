"""Command-line entry point for the face-mask detector."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from mask_detector import IMAGE_EXTS, VIDEO_EXTS, MaskDetector

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "mask_detector.pt"
VIDEOS_DIR = PROJECT_ROOT / "videos"
OUTPUT_DIR = PROJECT_ROOT / "output"


def parse_args():
    p = argparse.ArgumentParser(description="Face Mask Detection + Tracking")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--source", type=str, help="Path to one image or video file.")
    source.add_argument("--webcam", action="store_true", help="Run on the live webcam.")
    p.add_argument("--cam-index", type=int, default=0, help="Webcam device index (default: 0).")
    p.add_argument("--save", action="store_true", help="With --webcam, save annotated video to output/.")
    p.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to YOLO .pt weights.")
    p.add_argument("--conf", type=float, default=0.40, help="Confidence threshold, 0-1 (default: 0.40).")
    p.add_argument("--iou", type=float, default=0.50, help="IoU threshold (default: 0.50).")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640).")
    p.add_argument("--device", type=str, default=None, help="Inference device, e.g. cpu, 0, cuda:0.")
    p.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame.")
    p.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Ultralytics tracker config.")
    p.add_argument("--mask-class", type=str, default=None, help="Exact class name to force as Mask.")
    p.add_argument("--no-mask-class", type=str, default=None, help="Exact class name to force as No Mask.")
    p.add_argument("--show", action="store_true", help="Show a preview window while processing a file.")
    return p.parse_args()


def gather_inputs(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS.union(VIDEO_EXTS)
    )


def validate_range(name: str, value: float, low: float, high: float):
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}; got {value}.")


def main():
    args = parse_args()
    try:
        validate_range("--conf", args.conf, 0.0, 1.0)
        validate_range("--iou", args.iou, 0.0, 1.0)
        if args.imgsz <= 0 or args.max_det <= 0:
            raise ValueError("--imgsz and --max-det must be positive integers.")

        model_path = Path(args.model).expanduser().resolve()
        if not model_path.exists():
            print(f"[!] Model weights not found: {model_path}")
            print("    Run: python download_model.py")
            print("    Or pass: --model path/to/your/best.pt")
            return 1

        detector = MaskDetector(
            str(model_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
            mask_class=args.mask_class,
            no_mask_class=args.no_mask_class,
            tracker=args.tracker,
        )
        print(f"Loaded model: {model_path}")
        print(f"Classes: {detector.class_names}")

        if args.webcam:
            save_path = OUTPUT_DIR / "webcam_recording.mp4" if args.save else None
            detector.process_webcam(args.cam_index, str(save_path) if save_path else None)
            return 0

        targets = [Path(args.source).expanduser()] if args.source else gather_inputs(VIDEOS_DIR)
        if not targets:
            print(f"[!] No image/video files found in '{VIDEOS_DIR}'.")
            print("    Drop a test image/video into videos/ or run: python main.py --webcam")
            return 0

        failures = 0
        for path in targets:
            if not path.exists():
                print(f"[error] Source does not exist: {path}")
                failures += 1
                continue
            ext = path.suffix.lower()
            print(f"\nProcessing: {path}")
            try:
                if ext in IMAGE_EXTS:
                    out = detector.process_image(str(path), str(OUTPUT_DIR))
                elif ext in VIDEO_EXTS:
                    out = detector.process_video(str(path), str(OUTPUT_DIR), show=args.show)
                else:
                    raise ValueError(f"Unsupported source extension: {ext}")
                print(f"  -> saved: {out}")
            except Exception as exc:
                failures += 1
                print(f"  [error] {exc}")

        print(f"\nDone. Results are in '{OUTPUT_DIR}'.")
        return 1 if failures else 0
    except Exception as exc:
        print(f"[fatal] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
