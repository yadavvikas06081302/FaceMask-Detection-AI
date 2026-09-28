"""
Optional training placeholder for the Face Mask Detection AI project.

Dataset structure:
dataset/
  with_mask/
  without_mask/

For a full CNN training pipeline, install TensorFlow and prepare enough
labeled images. This file intentionally does not download a dataset.
"""
from pathlib import Path

ROOT = Path(__file__).parent
MASK_DIR = ROOT / "dataset" / "with_mask"
NO_MASK_DIR = ROOT / "dataset" / "without_mask"

print("Face Mask Detection AI - training setup")
print("With-mask images:", len(list(MASK_DIR.glob("*"))))
print("Without-mask images:", len(list(NO_MASK_DIR.glob("*"))))
print("\nAdd your labeled images to the two folders before implementing/training a CNN.")
print("The included app.py is ready to run without TensorFlow.")
