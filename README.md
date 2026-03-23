# Microfluidic subteam
# Urised2K Dataset Trained with YOLOv8

YOLO-based object detection project for urine sediment images.

This repository provides tools for:

- Converting XML annotations to YOLO format (.txt)
- Training a YOLOv8 model
- Running object detection on test images

---

Model performance on "evaluation" image after 5 epochs (1 hour on CPU)
Correctly identified 25 out of 27 ground truth labels
test image: 2014-05-17_S3_P107.jpg

Model performance on "evaluation" image after 50 epochs (20 mins T4 GPU)
Reached test mAP of 91.7% on all categories.

WBC 92%/ RBC 98%/ broken WBC 93.2%/ broken RBC 95.8%

train/val/test division 80/10/10

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
pip install ultralytics
```

---

# Training

Run the following command to start training:

```bash
python train.py
```

Training results will be saved to:

./runs/detect/train

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
