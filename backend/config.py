"""
Helmet Detection System - Backend Configuration
===============================================
Configuration settings for the Flask API and Deep Learning model.
"""

import os

class Config:
    # Server Settings
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    # Deep Learning Model Settings
    # Path to custom trained YOLO model weights (.pt or .onnx)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS", os.path.join(BASE_DIR, "weights", "helmet_model.pt"))
    
    # Default confidence threshold for detection filtering
    DEFAULT_CONFIDENCE_THRESHOLD = 0.65

    # Standard classes for helmet & safety compliance
    CLASS_LABELS = {
        0: "Helmet",
        1: "No Helmet",
        2: "Rider",
        3: "Motorcycle"
    }

    # Bounding box color codes (for optional server-side visualization)
    CLASS_COLORS = {
        "Helmet": "#10b981",       # Emerald green
        "No Helmet": "#f43f5e",    # Crimson red
        "Rider": "#3b82f6",        # Blue
        "Motorcycle": "#8b5cf6"    # Purple
    }
