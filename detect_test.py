"""
YOLOv8 Detection Script

This script loads a trained YOLO model (.pt file) and runs object detection
on a test image. The image path is entered by the user.

Usage:
    python detect_test.py
    then type the image path

Example:
    Enter image path: test/test.jpg
"""

from ultralytics import YOLO
from pathlib import Path


# =========================
# Model path
# =========================
MODEL_PATH = "./models/best.pt"


# =========================
# Output folder
# =========================
OUTPUT_PROJECT = "."
OUTPUT_NAME = "predict"


def main():

    # ask user for image path
    source_image = input("Enter image path: ").strip()

    # check image exists
    if not Path(source_image).exists():
        print("Error: Image file not found.")
        return

    # check model exists
    if not Path(MODEL_PATH).exists():
        print("Error: Model file not found.")
        return

    # load model
    model = YOLO(MODEL_PATH)

    # run prediction
    results = model.predict(
        source=source_image,
        save=True,
        project=OUTPUT_PROJECT,
        name=OUTPUT_NAME,
        exist_ok=True
    )

    print("\nDetection finished.")
    print("Input image:", source_image)
    print("Output saved to: ./predict")

    # print detection results
    for r in results:

        print("\nImage:", r.path)

        if r.boxes is None:
            print("No objects detected.")
            continue

        names = r.names

        for box in r.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            print(f"Detected: {names[cls_id]} | confidence={confidence:.3f}")


if __name__ == "__main__":
    main()