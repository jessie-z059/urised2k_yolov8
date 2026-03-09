# Urised YOLO

YOLO-based object detection project for urine sediment images.

This repository provides tools for:

- Converting XML annotations to YOLO format
- Training a YOLOv8 model
- Running object detection on test images

---

# Setup Environment

It is recommended to create a Python virtual environment before installing dependencies.

## 1. Create virtual environment

Mac / Linux

```bash
python3 -m venv venv
```

Windows

```bash
python -m venv venv
```
---

## 2. Activate virtual environment

Mac / Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Training

Run the following command to start training:

```bash
python train.py
```

Training results will be saved to:

```bash
./runs/detect/train
```

The best model will be stored at:

./runs/detect/train/weights/best.pt

After training, copy the trained model to the `models` directory before running inference.

---

# Inference

Run detection on a test image:

```bash
python detect_test.py
```

By default, detected images will be saved to:

predict/

The input image path and output directory can be modified in `detect_test.py`.

Example:

SOURCE_IMAGE = "test/test.jpg"

OUTPUT_PROJECT = "."
OUTPUT_NAME = "predict"

---

# Notes

- Dataset configuration is defined in `dataset.yaml`
- The base YOLO model used is `yolov8n.pt`
- Training outputs are automatically saved in the `runs` directory