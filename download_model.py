"""Download the default community face-mask YOLO checkpoint."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "models"
OUT_PATH = MODEL_DIR / "mask_detector.pt"
REPO_ID = "krishnamishra8848/Face_Mask_Detection"
FILENAME = "best.pt"


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        print(f"Model already exists: {OUT_PATH}")
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        print("huggingface_hub is missing. Install requirements first:")
        print("  python -m pip install -r requirements.txt")
        return 1

    print(f"Downloading {REPO_ID}/{FILENAME} ...")
    try:
        cached = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        shutil.copy2(cached, OUT_PATH)
    except Exception as exc:
        print(f"[!] Download failed: {exc}")
        print(f"Place a compatible YOLO .pt file at: {OUT_PATH}")
        return 1

    print(f"Saved -> {OUT_PATH}")
    try:
        from ultralytics import YOLO
        model = YOLO(str(OUT_PATH))
        print(f"Model loaded OK. Classes: {model.names}")
        print("Note: this community checkpoint does not publish benchmark metrics in its model card.")
    except Exception as exc:
        print(f"[!] Model downloaded but could not be loaded: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
