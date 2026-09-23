# 😷 FaceMask Detection AI — New Version

A clean, beginner-friendly face-mask detection project using **OpenCV + Keras/TensorFlow**.

## Features
- 📷 Live webcam detection
- 🖼️ Image detection
- 🎥 Video detection
- 👥 Multiple-face detection
- 📊 Mask / No Mask counters
- 💾 Save annotated images/videos
- ⬇️ Automatic model download
- ⚙️ Configurable confidence threshold
- 🧹 Clean project structure

> The classifier model is downloaded on first run from a public Hugging Face model repository. The project does **not** store the large model file inside the ZIP.

## 1. Install

Recommended: Python 3.10 or 3.11.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Then:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Download the model

```bash
python download_model.py
```

The model will be saved to:

```text
models/mask_detector.keras
```

## 3. Run webcam

```bash
python main.py --webcam
```

Press **Q** to quit and **S** to save a snapshot.

## 4. Run an image

```bash
python main.py --image path/to/photo.jpg
```

## 5. Run a video

```bash
python main.py --video path/to/video.mp4 --save
```

## Troubleshooting

### Camera does not open
Try:
```bash
python main.py --webcam --camera 1
```

### Model download fails
Run:
```bash
python download_model.py
```
again with an active internet connection.

### Labels appear reversed
The project supports:
```bash
python main.py --webcam --swap-labels
```

This is useful if a third-party model uses the opposite output order.

## Project structure

```text
FaceMask-Detection-AI-New/
├── models/
├── outputs/
├── main.py
├── mask_detector.py
├── download_model.py
├── requirements.txt
├── config.py
├── test_project.py
└── README.md
```

## Important note

This is an educational/computer-vision project. Face-mask predictions can be wrong because lighting, pose, occlusion and model limitations affect classification. Do not use predictions as the sole basis for safety-critical decisions.
