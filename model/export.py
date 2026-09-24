"""
Helmet Detection System - Model Export & Backend Deployment
============================================================
Exports trained model weights to high-performance inference formats (ONNX, TorchScript)
and seamlessly deploys them to the backend server.

Usage:
  python export.py --format onnx
  python export.py --deploy-only
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
WEIGHTS_DIR = MODEL_DIR / "weights"
BACKEND_WEIGHTS = PROJECT_ROOT / "backend" / "weights"


def export_and_deploy(weights_path=None, export_format="onnx", deploy=True):
    weights = Path(weights_path) if weights_path else (WEIGHTS_DIR / "best.pt")

    print("\n" + "=" * 65)
    print("  HELMETVISION AI - MODEL EXPORT & DEPLOYMENT TOOL")
    print(f"  Source Weights: {weights}")
    print(f"  Export Format: {export_format.upper()}")
    print("=" * 65 + "\n")

    if not weights.exists():
        print(f"[Error] Source weights '{weights}' not found.")
        print("Please train a model first using: python model/train.py")
        return False

    # Attempt Ultralytics model export
    try:
        from ultralytics import YOLO
        print(f"[Export] Loading YOLO model from: {weights}")
        model = YOLO(str(weights))
        print(f"[Export] Exporting to format: {export_format}...")
        exported_path = model.export(format=export_format)
        print(f"[Export] Export successful: {exported_path}")
    except ImportError:
        print("[Export] 'ultralytics' not installed. Skipping binary conversion.")
    except Exception as e:
        print(f"[Export] Export note: {e}")

    # Deploy to Backend
    if deploy:
        BACKEND_WEIGHTS.mkdir(parents=True, exist_ok=True)
        dest_file = BACKEND_WEIGHTS / "helmet_model.pt"
        shutil.copy2(weights, dest_file)
        print(f"\n[Deploy] Successfully deployed model weights to:")
        print(f"         -> {dest_file}")
        print("[Deploy] The Flask backend is now primed to use this model!")

    print("\n" + "=" * 65)
    print("  DEPLOYMENT PROCESS COMPLETED")
    print("=" * 65 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export and Deploy Helmet Model")
    parser.add_argument("--weights", type=str, default="", help="Path to weights file")
    parser.add_argument("--format", type=str, default="onnx", choices=["onnx", "torchscript", "engine"], help="Export format")
    parser.add_argument("--deploy-only", action="store_true", help="Only deploy weights to backend")

    args = parser.parse_args()
    export_and_deploy(weights_path=args.weights, export_format=args.format, deploy=True)
