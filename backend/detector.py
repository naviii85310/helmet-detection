"""
Helmet Detection System - Deep Learning Inference Engine
=========================================================
Handles image decoding, YOLO/PyTorch model execution, bounding box scaling,
and safety compliance classification.
"""

import base64
import io
import math
import os
import time
from backend.config import Config

class HelmetDetector:
    def __init__(self, weights_path=None):
        self.weights_path = weights_path or Config.MODEL_WEIGHTS_PATH
        self.model = None
        self.mode = "fallback"  # 'yolo' or 'fallback'
        self.frame_counter = 0

        self._initialize_model()

    def _initialize_model(self):
        """Attempts to load a YOLO model if ultralytics and weights are available."""
        if os.path.exists(self.weights_path):
            try:
                from ultralytics import YOLO
                print(f"[Detector] Loading YOLO model from: {self.weights_path}")
                self.model = YOLO(self.weights_path)
                self.mode = "yolo"
                print("[Detector] Successfully loaded custom YOLO model.")
                return
            except ImportError:
                print("[Detector] 'ultralytics' not installed. Running in heuristic/simulation fallback mode.")
            except Exception as e:
                print(f"[Detector] Error loading weights: {e}. Falling back to simulation mode.")
        else:
            print(f"[Detector] Model weights not found at '{self.weights_path}'.")
            print("[Detector] Operating in DL Simulation/Heuristic mode.")
            print("[Detector] Tip: Place your trained YOLO .pt weights in 'backend/weights/' to enable live YOLO inference.")

    def decode_image(self, base64_str):
        """
        Decodes a Base64 encoded JPEG/PNG image string sent by the frontend.
        Returns the decoded image (as OpenCV BGR array or PIL Image) and dimensions (width, height).
        """
        if not base64_str:
            return None, 0, 0

        # Remove data URI prefix if present
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]

        image_bytes = base64.b64decode(base64_str)

        # Attempt OpenCV decode
        try:
            import cv2
            import numpy as np
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                return img, w, h
        except ImportError:
            pass

        # Fallback to PIL (Pillow)
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(image_bytes))
            w, h = img.size
            return img, w, h
        except ImportError:
            pass

        # Minimal fallback if neither cv2 nor PIL is installed: return raw bytes with default 640x360 dimensions
        return image_bytes, 640, 360

    def detect(self, base64_str, threshold=None):
        """
        Executes helmet detection inference on the input image.
        Returns a dictionary containing detection bounding boxes, labels, and timings.
        """
        start_time = time.time()
        threshold = threshold or Config.DEFAULT_CONFIDENCE_THRESHOLD

        img, img_w, img_h = self.decode_image(base64_str)
        if img is None:
            return {
                "status": "error",
                "message": "Invalid or unreadable image data",
                "detections": [],
                "inference_time_ms": 0
            }

        self.frame_counter += 1

        if self.mode == "yolo" and self.model is not None:
            detections = self._run_yolo_inference(img, img_w, img_h, threshold)
        else:
            detections = self._run_heuristic_inference(img_w, img_h, threshold)

        inference_time_ms = int((time.time() - start_time) * 1000)

        # Categorize overall safety status
        has_violation = any(d.get("type") == "violation" for d in detections)
        has_helmet = any(d.get("type") == "helmet" for d in detections)

        if has_violation:
            overall_status = "violation"
        elif has_helmet:
            overall_status = "safe"
        else:
            overall_status = "idle"

        return {
            "status": "success",
            "model_mode": self.mode,
            "overall_status": overall_status,
            "detections": detections,
            "image_width": img_w,
            "image_height": img_h,
            "inference_time_ms": inference_time_ms
        }

    def _run_yolo_inference(self, img, img_w, img_h, threshold):
        """Runs actual YOLO inference using the loaded model."""
        results = self.model.predict(img, conf=threshold, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < threshold:
                    continue

                label = self.model.names.get(cls_id, f"Class {cls_id}")
                label_lower = label.lower()

                # Determine compliance type
                if "no" in label_lower or "violation" in label_lower:
                    det_type = "violation"
                elif "helmet" in label_lower:
                    det_type = "helmet"
                else:
                    det_type = "person"

                # Box coordinates: [x1, y1, x2, y2] -> convert to [x, y, width, height]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bx = max(0, round(x1))
                by = max(0, round(y1))
                bw = min(img_w - bx, round(x2 - x1))
                bh = min(img_h - by, round(y2 - y1))

                detections.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "box": [bx, by, bw, bh],
                    "type": det_type
                })

        return detections

    def _run_heuristic_inference(self, img_w, img_h, threshold):
        """
        Realistic heuristic & simulation detector:
        Generates realistic smooth bounding boxes for riders and helmets/no-helmets,
        allowing testing and frontend validation even before custom YOLO weights are trained.
        """
        w = max(img_w, 320)
        h = max(img_h, 240)

        # Subtle dynamic movement over frames
        t = self.frame_counter * 0.08
        offset_x = math.sin(t) * (w * 0.04)
        offset_y = math.cos(t * 0.7) * (h * 0.02)

        # Rider box (central area)
        rider_w = round(w * 0.35)
        rider_h = round(h * 0.65)
        rider_x = round((w - rider_w) / 2 + offset_x)
        rider_y = round(h * 0.2 + offset_y)

        # Head / Helmet box (upper part of rider)
        head_w = round(rider_w * 0.52)
        head_h = round(rider_h * 0.3)
        head_x = round(rider_x + (rider_w - head_w) / 2)
        head_y = round(rider_y + (rider_h * 0.03))

        # Alternate every few cycles between compliant helmet (80% time) and violation (20% time)
        cycle = (self.frame_counter // 40) % 5
        is_violation = (cycle == 3)

        detections = [
            {
                "label": "Motorcyclist",
                "confidence": 0.96,
                "box": [rider_x, rider_y, rider_w, rider_h],
                "type": "person"
            }
        ]

        if is_violation:
            conf = round(0.88 + (math.sin(t) * 0.04), 2)
            if conf >= threshold:
                detections.append({
                    "label": "No Helmet",
                    "confidence": conf,
                    "box": [head_x, head_y, head_w, head_h],
                    "type": "violation"
                })
        else:
            conf = round(0.94 + (math.cos(t) * 0.03), 2)
            if conf >= threshold:
                detections.append({
                    "label": "Helmet",
                    "confidence": conf,
                    "box": [head_x, head_y, head_w, head_h],
                    "type": "helmet"
                })

        return detections
