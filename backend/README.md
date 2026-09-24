# Helmet Detection System - Backend (Flask & Deep Learning)

This directory contains the Python Flask REST API and Deep Learning inference engine for the **HelmetVision AI** system.

---

## Directory Structure

```
backend/
├── app.py              # Main Flask server entrypoint (defines REST API routes)
├── detector.py         # Deep Learning detection engine (YOLO / PyTorch + fallback)
├── config.py           # Server & model configuration settings
├── requirements.txt    # Python dependencies
├── weights/            # Directory to store trained model weights (.pt files)
└── README.md           # Documentation and setup instructions
```

---

## Quick Start Guide

### 1. Create a Virtual Environment (Recommended)
Open a terminal in the project directory:
```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate
# (Or on Linux/macOS: source venv/bin/activate)
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Backend Server
```bash
python app.py
```
The server will start at:
- **Local URL**: `http://127.0.0.1:5000`
- **Inference Route**: `http://127.0.0.1:5000/detect`

---

## Connecting with Custom YOLO Weights

1. Create a folder named `weights` inside `backend/`:
   ```bash
   mkdir backend/weights
   ```
2. Place your trained YOLO model (e.g. `helmet_model.pt` or `best.pt`) inside `backend/weights/helmet_model.pt`.
3. If your weights file has a different name, either rename it or update `MODEL_WEIGHTS_PATH` in `backend/config.py`.
4. When `ultralytics` and your `.pt` file are detected, `detector.py` will automatically switch to **YOLO inference mode**.

---

## API Endpoints

### 1. `POST /detect`
Accepts a video frame from the camera as a Base64-encoded string and runs helmet detection.

**Request Payload:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "threshold": 0.65
}
```

**Response Format:**
```json
{
  "status": "success",
  "model_mode": "yolo",
  "overall_status": "safe",
  "detections": [
    {
      "label": "Helmet",
      "confidence": 0.94,
      "box": [220, 110, 160, 180],
      "type": "helmet"
    },
    {
      "label": "Motorcyclist",
      "confidence": 0.97,
      "box": [180, 100, 240, 480],
      "type": "person"
    }
  ],
  "image_width": 640,
  "image_height": 360,
  "inference_time_ms": 28
}
```

### 2. `POST /ping`
Tests backend connectivity from the frontend's Settings modal.

**Response:**
```json
{
  "status": "success",
  "message": "HelmetVision Backend is reachable and responsive!",
  "engine_mode": "fallback"
}
```

### 3. `GET /health`
Returns health check status and total processed frames.

---

## Connecting Frontend to Backend

1. Start this backend server: `python backend/app.py`.
2. Open the frontend in your browser: `frontend/index.html`.
3. Click **"Backend Config"** in the top navigation bar.
4. Select **"Live Python Backend (Flask / FastAPI)"**.
5. Ensure the API endpoint is set to `http://127.0.0.1:5000/detect`.
6. Click **"Test Ping"** to verify connection, then click **"Save Configuration"**.
7. Start your camera — your frontend will now transmit live frames to this Flask backend!
