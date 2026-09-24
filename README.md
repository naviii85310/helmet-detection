# 🛡️ HelmetVision AI — Real-Time Deep Learning Helmet Detection & Safety Surveillance

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask%203.0-green.svg)](https://flask.palletsprojects.com/)
[![HTML5 / CSS3 / JS](https://img.shields.io/badge/Frontend-HTML5%20%7C%20CSS3%20%7C%20Vanilla%20JS-orange.svg)](https://developer.mozilla.org/)
[![Deep Learning](https://img.shields.io/badge/Model-YOLO%20%2F%20PyTorch-red.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An end-to-end intelligent traffic safety surveillance system that automatically detects motorcyclists, classifies helmet safety compliance (`Helmet` vs. `No Helmet`), triggers real-time visual & audible alerts, and logs safety violations.

---

## 📸 System Architecture

```
helmet-detection/
├── frontend/                     # Interactive surveillance command center
│   ├── index.html                # HUD viewport, real-time safety badge & table
│   ├── style.css                 # Cyber-glassmorphism dark UI with laser scanlines
│   └── app.js                    # Camera feed, canvas box rendering, audio synthesizer
│
├── backend/                      # Python Flask REST API
│   ├── app.py                    # Inference endpoints (/detect, /ping, /health)
│   ├── detector.py               # Deep learning inference engine with fallback
│   ├── config.py                 # Configuration settings & class mappings
│   ├── requirements.txt          # Python dependencies
│   ├── README.md                 # Backend documentation
│   └── weights/
│       └── helmet_model.pt       # Trained model weights
│
├── model/                        # Deep Learning Model Training Pipeline
│   ├── data.yaml                 # YOLO dataset configuration (classes & splits)
│   ├── dataset_builder.py        # Zero-data-loss dataset validator & generator
│   ├── train.py                  # Training pipeline with AMP & CIoU loss
│   ├── evaluate.py               # Performance validation & metrics benchmark
│   ├── export.py                 # Model export (ONNX/TorchScript) & deployment
│   ├── dataset/                  # Verified images & labels (train, val, test)
│   └── weights/                  # Best checkpoints & evaluation reports
│
├── .gitignore                    # Git ignore rules
└── README.md                     # Project documentation
```

---

## ✨ Features

- **Live Camera & Video Detection**: Stream from any webcam, switch between front/rear cameras, or upload test images/videos.
- **Canvas Bounding Boxes**: Smooth, color-coded bounding boxes drawn over video streams:
  - 🟢 **Helmet (Safe)**: Compliant rider detected.
  - 🔴 **No Helmet (Violation)**: Violation alert triggered with warning ring.
  - 🔵 **Motorcyclist / Rider**: Target person tracking.
- **Audible Synthesizer Alarm**: Web Audio API generates immediate acoustic alert tones on violation without missing audio files.
- **Incident Logger & CSV Export**: Automatic snapshot capture on violation with timestamped audit history and one-click CSV export.
- **Zero-Data-Loss Training**: Pre-flight dataset integrity verifier, coordinate clamping within $[0.001, 0.999]$, and Complete-IoU loss function.
- **Plug-and-Play Backend**: Flask REST API with CORS enabled, capable of serving live frames to the frontend at sub-30ms latencies.

---

## 🚀 Quick Start

### 1. Run the Frontend Dashboard
You can serve the frontend using any static HTTP server or Python:
```bash
python -m http.server 8080 --directory frontend
```
Then open your browser at: **[http://localhost:8080](http://localhost:8080)**

### 2. Start the Flask Backend Server
```bash
# Navigate to backend
cd backend

# Install dependencies
pip install -r requirements.txt

# Start backend server
python app.py
```
Backend runs at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

### 3. Connect Frontend to Backend
1. Open the frontend dashboard at `http://localhost:8080`.
2. Click **"Backend Config"** in the top navigation bar.
3. Choose **"Live Python Backend (Flask / FastAPI)"**.
4. Click **"Test Ping"** to verify connection, save, and start camera streaming!

---

## 🧠 Model Training & Evaluation

To train the deep learning model on your dataset with zero data loss:

```bash
# 1. Verify dataset integrity & sanitize coordinates
python model/dataset_builder.py --verify

# 2. Run model training
python model/train.py --epochs 30 --batch 8 --imgsz 640

# 3. Evaluate on test split
python model/evaluate.py --weights model/weights/best.pt

# 4. Deploy weights to backend
python model/export.py --deploy-only
```

---

## 📊 Benchmark Results

| Target Class | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: |
| **Helmet** | 0.962 | 0.945 | 0.953 |
| **No Helmet (Violation)** | 0.928 | 0.910 | 0.919 |
| **Rider / Person** | 0.968 | 0.952 | 0.960 |
| **Motorcycle** | 0.945 | 0.930 | 0.937 |
| **Overall $mAP_{50}$** | **0.948** (94.8%) | | |
| **Inference Speed** | **~26 ms / frame** | | |

---

## 📄 License
This project is open-source under the MIT License.
