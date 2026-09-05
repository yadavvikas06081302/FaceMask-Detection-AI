"""Evaluate a YOLO face-mask model on a labeled YOLO dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Validate a face-mask YOLO model.")
    p.add_argument("--model", required=True, help="Path to model weights (.pt).")
    p.add_argument("--data", required=True, help="Path to YOLO dataset data.yaml.")
    p.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split (default: val).")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001, help="Validation confidence threshold; 0.001 is a common recall-friendly setting.")
    p.add_argument("--device", default=None)
    p.add_argument("--output", default="output/metrics.json")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is not installed. Run: python -m pip install -r requirements.txt") from exc

    model = YOLO(args.model)
    kwargs = {
        "data": args.data,
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "verbose": False,
    }
    if args.device:
        kwargs["device"] = args.device
    metrics = model.val(**kwargs)

    box = metrics.box
    payload = {
        "model": str(Path(args.model).resolve()),
        "data": str(Path(args.data).resolve()),
        "split": args.split,
        "imgsz": args.imgsz,
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "classes": model.names,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"mAP50:    {payload['mAP50']:.4f}")
    print(f"mAP50-95: {payload['mAP50_95']:.4f}")
    print(f"Precision:{payload['precision']:.4f}")
    print(f"Recall:   {payload['recall']:.4f}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
