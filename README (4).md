# 😷 FaceMask Detection AI

<p align="center">
  <strong>Real-Time Face Mask Detection using YOLO, OpenCV & ByteTrack</strong>
</p>

<p align="center">
  An end-to-end computer vision application for detecting face-mask usage from images, videos, and live webcam streams.
</p>

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-FF6F00?style=for-the-badge)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)](https://pytorch.org/)
[![ByteTrack](https://img.shields.io/badge/ByteTrack-Multi--Object%20Tracking-8A2BE2?style=for-the-badge)](https://github.com/ifzhang/ByteTrack)

</p>

<p align="center">

[![GitHub stars](https://img.shields.io/github/stars/HadeedJalani/FaceMask-Detection-AI?style=flat-square)](https://github.com/HadeedJalani/FaceMask-Detection-AI/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/HadeedJalani/FaceMask-Detection-AI?style=flat-square)](https://github.com/HadeedJalani/FaceMask-Detection-AI/network/members)
[![GitHub issues](https://img.shields.io/github/issues/HadeedJalani/FaceMask-Detection-AI?style=flat-square)](https://github.com/HadeedJalani/FaceMask-Detection-AI/issues)
[![GitHub last commit](https://img.shields.io/github/last-commit/HadeedJalani/FaceMask-Detection-AI?style=flat-square)](https://github.com/HadeedJalani/FaceMask-Detection-AI/commits/main)

</p>

---

## 📌 Overview

**FaceMask Detection AI** is a computer vision project designed to detect whether people are wearing face masks using a YOLO-based object detection model.

The application supports:

* 🖼️ Image detection
* 🎥 Video detection
* 📷 Real-time webcam detection
* 🛰️ Multi-object tracking with ByteTrack
* 📊 Live detection statistics
* 🎯 Configurable confidence thresholds
* ⚙️ CPU and GPU inference
* 📈 Model performance evaluation
* 🧪 Automated testing
* 🔄 GitHub Actions continuous integration

The project is designed to be easy to run locally while maintaining a clean structure suitable for further development, experimentation, and deployment.

---

# ✨ Key Features

### 🎯 YOLO-Based Detection

Uses a YOLO object detection model to identify face-mask-related classes in visual input.

### 📷 Real-Time Webcam Detection

Run the detector directly against your webcam for real-time monitoring.

```bash
python main.py --webcam
```

### 🎥 Video Processing

Process existing videos and generate annotated detection output.

```bash
python main.py --source path/to/video.mp4
```

### 🛰️ ByteTrack Tracking

For video and webcam streams, ByteTrack can maintain persistent IDs across frames.

Example:

```text
Frame 1  → Person → Track ID 4
Frame 2  → Person → Track ID 4
Frame 3  → Person → Track ID 4
```

This makes it possible to track the same detected person across consecutive frames.

### 📊 Live Statistics

The application maintains live detection/tracking information instead of simply accumulating detections from previous frames.

### ⚙️ Flexible Inference

Configure:

* Detection confidence
* Image size
* CPU/GPU device
* Model checkpoint
* Input source
* Display/output behavior

### 📈 Evaluation

The project includes an evaluation script capable of reporting:

* Precision
* Recall
* mAP@50
* mAP@50-95

This provides a more meaningful measurement of model performance than an unsupported single "accuracy %" number.

---

# 🧠 Detection Pipeline

```text
                    ┌───────────────────┐
                    │       INPUT       │
                    │                   │
                    │ Image / Video /   │
                    │      Webcam       │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   YOLO DETECTOR   │
                    │                   │
                    │ Object Detection  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   CLASS MAPPING   │
                    │                   │
                    │ Mask / No Mask /  │
                    │ Other Model Class │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │    BYTE TRACK     │
                    │                   │
                    │ Persistent IDs    │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ ANNOTATED OUTPUT  │
                    │                   │
                    │ Labels + Boxes +  │
                    │ Tracking Info     │
                    └───────────────────┘
```

---

# 🛠️ Tech Stack

| Technology              | Purpose                 |
| ----------------------- | ----------------------- |
| 🐍 **Python**           | Application development |
| 🎯 **Ultralytics YOLO** | Object detection        |
| 🔥 **PyTorch**          | Deep learning framework |
| 👁️ **OpenCV**          | Image/video processing  |
| 🛰️ **ByteTrack**       | Multi-object tracking   |
| 🔢 **NumPy**            | Numerical operations    |
| 🧪 **unittest**         | Automated testing       |
| 🤖 **GitHub Actions**   | Continuous integration  |

---

# 📂 Project Structure

```text
FaceMask-Detection-AI/
│
├── 📁 models/
│   └── mask_detector.pt
│
├── 📁 dataset/
│   └── data.yaml
│
├── 📁 tests/
│   └── test_pipeline.py
│
├── 📁 output/
│   └── generated results
│
├── 📁 .github/
│   └── 📁 workflows/
│       └── tests.yml
│
├── 🐍 main.py
├── 📊 evaluate.py
├── ⬇️ download_model.py
├── 📓 colab_train_yolov11_mask_detector.ipynb
├── 📋 requirements.txt
├── 📜 NOTICE.md
├── 📖 README.md
└── 🚫 .gitignore
```

> **Note:** Model weights and generated output files should generally not be committed to Git unless their redistribution and licensing terms have been verified.

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/HadeedJalani/FaceMask-Detection-AI.git
cd FaceMask-Detection-AI
```

---

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell reports that script execution is disabled:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

You should see:

```text
(.venv) PS C:\...\FaceMask-Detection-AI>
```

---

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

---

# 📦 Install Dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

---

# 🤖 Model Setup

Download the required model:

```bash
python download_model.py
```

The application expects the model at:

```text
models/mask_detector.pt
```

You can also provide your own compatible YOLO model:

```bash
python main.py --model path/to/your/best.pt
```

---

# ▶️ Usage

## 🖼️ Image Detection

```bash
python main.py --source path/to/image.jpg
```

Example:

```bash
python main.py --source test.jpg
```

---

## 🎥 Video Detection

```bash
python main.py --source path/to/video.mp4
```

Example:

```bash
python main.py --source test.mp4
```

---

## 📷 Webcam Detection

Start real-time webcam detection:

```bash
python main.py --webcam
```

Save the webcam output:

```bash
python main.py --webcam --save
```

---

# ⚙️ Configuration

### Confidence Threshold

```bash
python main.py --conf 0.40
```

Higher values generally produce fewer but more confident detections.

---

### Image Size

```bash
python main.py --imgsz 640
```

---

### CPU Inference

```bash
python main.py --device cpu
```

---

### GPU Inference

If a compatible CUDA environment is available:

```bash
python main.py --device 0
```

---

### Display Results

```bash
python main.py --show
```

---

### Combined Example

```bash
python main.py --source test.mp4 --conf 0.40 --imgsz 640 --device 0 --show
```

---

# 🏷️ Class Mapping

Different trained models can use different class names and class ordering.

For this reason, the application supports explicit class mapping.

Example:

```bash
python main.py \
  --model models/best.pt \
  --mask-class with_mask \
  --no-mask-class without_mask
```

This helps prevent a dangerous situation where the numerical class index is correct but the application displays the wrong semantic label.

---

# 🛰️ Tracking with ByteTrack

For video and webcam inputs, the application can use ByteTrack to associate detections between frames.

Conceptually:

```text
┌──────────────┐
│    Frame 1   │
│              │
│ Person #3    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Frame 2   │
│              │
│ Person #3    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Frame 3   │
│              │
│ Person #3    │
└──────────────┘
```

Tracking state is reset when a new input source begins.

---

# 📊 Model Evaluation

The project includes an evaluation script for measuring model performance on a labeled dataset.

Run:

```bash
python evaluate.py --model models/best.pt --data dataset/data.yaml --split val
```

The evaluation reports:

```text
Precision
Recall
mAP@50
mAP@50-95
```

Results are saved to:

```text
output/metrics.json
```

## Why mAP Instead of Simple Accuracy?

Object detection is not adequately represented by ordinary classification accuracy.

A proper detection benchmark should consider:

* Whether the object was detected
* Whether the predicted class was correct
* Whether the bounding box overlaps the ground truth
* Precision
* Recall
* Intersection over Union (IoU)
* mAP at different IoU thresholds

Therefore, this project avoids claiming a performance percentage without an actual benchmark dataset.

---

# 🧪 Testing

Run the automated test suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

A successful run should report all tests passing.

The project also includes GitHub Actions so tests can automatically run when changes are pushed to the repository.

---

# 📓 Model Training

A training notebook is included:

```text
colab_train_yolov11_mask_detector.ipynb
```

The notebook can be used as a starting point for training or experimenting with your own face-mask detection dataset.

For reproducible research, record:

* Dataset version
* Training/validation split
* Model checkpoint
* Image size
* Number of epochs
* Confidence threshold
* Evaluation metrics

---

# 📈 Improving Detection Accuracy

Detection performance depends heavily on the training dataset.

For better real-world performance, the dataset should contain variation in:

* Lighting
* Camera quality
* Face orientation
* Occlusion
* Mask styles
* Indoor/outdoor environments
* Single and multiple people
* Different distances from the camera

A model should be evaluated on data that was **not used for training or tuning**.

---

# 🔐 Security & Repository Hygiene

Do not commit:

```text
.env
API keys
passwords
private datasets
personal data
large generated outputs
unverified model weights
.venv/
Python caches
```

The repository includes a `.gitignore` to help keep local/environment-specific files out of version control.

---

# 📜 Third-Party Software

This project uses third-party libraries and model assets.

Important dependencies include:

* Ultralytics
* PyTorch
* OpenCV
* ByteTrack

Please review the applicable licenses before redistributing or using the project commercially.

See:

```text
NOTICE.md
```

for additional third-party attribution information.

---

# 🔮 Future Improvements

Potential future development includes:

* 🌐 Streamlit web interface
* ⚡ FastAPI inference API
* 📊 Real-time analytics dashboard
* 📁 CSV/JSON detection export
* 🔔 Configurable alerts
* 🧠 Improved custom-trained models
* 📱 Mobile/web deployment
* 🐳 Docker support
* ☁️ Cloud deployment
* 📈 Advanced performance benchmarking
* 🎥 FPS and latency monitoring

---

# 🤝 Contributing

Contributions, ideas, bug reports, and improvements are welcome.

### Recommended workflow

```bash
git checkout -b feature/my-feature
```

Make your changes, test them:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Commit:

```bash
git add .
git commit -m "Add my feature"
```

Push:

```bash
git push origin feature/my-feature
```

Then open a Pull Request.

---

# ⭐ Support the Project

If you find this project useful:

* ⭐ Star the repository
* 🍴 Fork the project
* 🐛 Report issues
* 💡 Suggest improvements
* 🤝 Contribute

Your support is appreciated!

---

# 👨‍💻 Author

## Hadeed Jalani

Computer Vision & AI Project

<p align="left">

**GitHub:** [@HadeedJalani](https://github.com/HadeedJalani)

**Repository:** [FaceMask-Detection-AI](https://github.com/HadeedJalani/FaceMask-Detection-AI)

</p>

---

<p align="center">
  <strong>Built with Python • YOLO • PyTorch • OpenCV • ByteTrack</strong>
</p>

<p align="center">
  Made by <strong>Hadeed Jalani</strong>
</p>
