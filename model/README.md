# Helmet Detection System - Deep Learning Model Pipeline

This directory contains the complete Deep Learning pipeline for training, evaluating, and exporting motorcycle helmet safety detection models with **strict Zero-Data-Loss guarantees**.

---

## Directory Structure

```
model/
├── data.yaml             # YOLO dataset configuration (classes & splits)
├── dataset_builder.py    # Zero-data-loss dataset preparation & validator
├── train.py              # Loss-free training pipeline (CIoU + BCE loss, AMP)
├── evaluate.py           # Model validation (mAP@50, mAP@50-95, precision, recall)
├── export.py             # Model export (ONNX, TorchScript) and backend deployment
├── dataset/              # Training, validation, and testing images/labels
│   ├── images/{train,val,test}
│   └── labels/{train,val,test}
├── weights/              # Saved model checkpoints and evaluation reports
│   ├── best.pt
│   ├── last.pt
│   ├── model_metadata.json
│   └── evaluation_report.json
└── README.md             # This guide
```

---

## Zero Data Loss Guarantees

During deep learning training, dataset loss often occurs due to:
1. Corrupted/truncated image files causing unhandled loader crashes.
2. Floating-point coordinate drift resulting in bounding boxes being silently dropped.
3. Class imbalance causing the model to miss small helmet bounding boxes.
4. Numerical underflow during backpropagation (NaN gradients).

### How this pipeline prevents data loss:
- **Pre-flight Dataset Audit**: `dataset_builder.py --verify` scans all images, checks byte lengths, sanitizes invalid annotations, and clamps bounding box coordinates into $[0.001, 0.999]$ normalized space without discarding labels.
- **Automatic Mixed Precision (AMP)**: `train.py` utilizes gradient scaling to eliminate underflow.
- **Complete-IoU (CIoU) Loss**: Accurately handles small object localization (helmets) by accounting for overlap area, center distance, and aspect ratio.
- **Continuous Checkpointing**: Saves `best.pt` and `last.pt` at every epoch with full recovery support (`--resume`).

---

## Step-by-Step Workflow

### 1. Prepare or Verify Dataset
To generate a verified starter benchmark dataset:
```bash
python model/dataset_builder.py --generate --samples 60
```
Or to verify an existing dataset:
```bash
python model/dataset_builder.py --verify
```

### 2. Import External Datasets (Roboflow / Kaggle)
If you have a downloaded helmet detection dataset:
```bash
python model/dataset_builder.py --import-dir /path/to/downloaded/dataset
```

### 3. Train the Model
```bash
# Basic training
python model/train.py --epochs 30 --batch 8 --imgsz 640

# Resume interrupted training
python model/train.py --resume
```

### 4. Evaluate Performance
```bash
python model/evaluate.py --weights model/weights/best.pt
```
This generates:
- Overall $mAP_{50}$ and $mAP_{50-95}$
- Per-class Precision, Recall, and F1-Scores
- Saved report in `model/weights/evaluation_report.json`

### 5. Deploy to Backend Server
```bash
python model/export.py --deploy-only
```
This copies `best.pt` directly into `backend/weights/helmet_model.pt`. When you launch `python backend/app.py`, the backend automatically connects to this trained model!
