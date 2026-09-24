"""
Helmet Detection System - Flask Backend REST API
=================================================
Provides endpoints for real-time video frame inference, health checking,
and DL model telemetry.

Usage:
    python app.py
"""

import sys
import os

# Ensure the parent directory is in sys.path so backend module imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from flask import Flask, request, jsonify
from backend.config import Config
from backend.detector import HelmetDetector

# Initialize Flask application
app = Flask(__name__)
app.config.from_object(Config)

# Enable CORS (Cross-Origin Resource Sharing)
try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}})
except ImportError:
    # Manual fallback CORS headers if flask_cors is not yet installed
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        return response

# Initialize the Deep Learning Detector
detector = HelmetDetector()


@app.route("/", methods=["GET"])
def index():
    """Root endpoint: returns system information and model status."""
    return jsonify({
        "system": "HelmetVision AI Surveillance Backend",
        "status": "online",
        "version": "1.0.0",
        "engine_mode": detector.mode,
        "weights_configured": detector.weights_path,
        "endpoints": {
            "detect": "POST /detect",
            "ping": "POST /ping",
            "health": "GET /health"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for container / orchestrator probes."""
    return jsonify({
        "status": "healthy",
        "engine_mode": detector.mode,
        "frames_processed": detector.frame_counter
    })


@app.route("/ping", methods=["POST", "GET"])
def ping():
    """Diagnostic ping route invoked by the frontend 'Test Ping' button."""
    return jsonify({
        "status": "success",
        "message": "HelmetVision Backend is reachable and responsive!",
        "engine_mode": detector.mode
    })


@app.route("/detect", methods=["POST", "OPTIONS"])
def detect():
    """
    Main inference route:
    Accepts Base64 image payload or multipart image from the frontend camera stream,
    runs detection, and returns bounding box coordinates and safety classification.
    """
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200

    data = request.get_json(silent=True) or {}

    # Support testing ping payload inside /detect
    if data.get("ping"):
        return jsonify({
            "status": "ok",
            "message": "Ping received on /detect endpoint."
        })

    # 1. Retrieve base64 image string
    image_base64 = data.get("image")

    # If multipart file was uploaded instead of JSON base64:
    if not image_base64 and "file" in request.files:
        import base64
        file_bytes = request.files["file"].read()
        image_base64 = base64.b64encode(file_bytes).decode("utf-8")
    elif not image_base64 and "image" in request.files:
        import base64
        file_bytes = request.files["image"].read()
        image_base64 = base64.b64encode(file_bytes).decode("utf-8")

    if not image_base64:
        return jsonify({
            "status": "error",
            "message": "Missing 'image' payload. Send base64 data or multipart file."
        }), 400

    # 2. Retrieve optional confidence threshold
    threshold = data.get("threshold")
    if threshold is not None:
        try:
            threshold = float(threshold)
        except (ValueError, TypeError):
            threshold = Config.DEFAULT_CONFIDENCE_THRESHOLD

    # 3. Execute Detection Inference
    result = detector.detect(image_base64, threshold=threshold)

    return jsonify(result)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  HELMETVISION AI - DEEP LEARNING BACKEND SERVER")
    print(f"  Server URL: http://{Config.HOST}:{Config.PORT}")
    print(f"  Inference Endpoint: http://127.0.0.1:{Config.PORT}/detect")
    print(f"  Detector Engine Mode: {detector.mode.upper()}")
    print("=" * 65 + "\n")

    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
