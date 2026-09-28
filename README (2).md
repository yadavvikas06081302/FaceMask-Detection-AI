# Face Mask Detection AI

**Author / Developer:** Vikas Yadav

A simple Python + OpenCV computer-vision project that opens the webcam,
detects faces, and displays a lightweight **MASK / NO MASK** estimate.

## Features

- Real-time webcam detection
- Face detection with OpenCV Haar Cascade
- Simple, lightweight setup
- No external model download required to run the demo
- Easy to modify and extend with a trained CNN model

## Project Structure

```text
FaceMask-Detection-AI/
├── app.py
├── mask_detector.py
├── train.py
├── requirements.txt
├── README.md
├── models/
└── dataset/
    ├── with_mask/
    └── without_mask/
```

## Requirements

- Windows / Linux / macOS
- Python 3.9 or newer
- Working webcam

## Run on Laptop

### 1. Open Terminal / Command Prompt

Go inside the project folder:

```bash
cd FaceMask-Detection-AI
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

If `pip` does not work, try:

```bash
python -m pip install -r requirements.txt
```

### 3. Start the project

```bash
python app.py
```

The webcam window will open.

### 4. Stop

Press **Q** inside the camera window.

## Important Note

The included version is a **lightweight runnable demo classifier**, not a medically
validated mask classifier. For an academic/production-quality system, replace
the heuristic in `app.py` with a trained CNN/transfer-learning model using a
proper labeled mask dataset.

## Adding Your Own Dataset

Put images here:

```text
dataset/with_mask/
dataset/without_mask/
```

Then use `train.py` as the starting point for your own model-training pipeline.

## Author

**Vikas Yadav**

GitHub: https://github.com/yadavvikas06081302

## License

For educational and project demonstration purposes.
