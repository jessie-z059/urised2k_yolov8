"""
YOLOv8 Detection Script

This script loads a trained YOLO model (.pt file) and runs object detection
on a test image.

Usage:
    python detect_test.py

Example:
    Enter image path: test/test.jpg
"""

from ultralytics import YOLO
from pathlib import Path


# =========================
# Base directory (project root)
# =========================
BASE_DIR = Path(__file__).parent


# =========================
# Model path
# =========================
MODEL_PATH = BASE_DIR / "models/best.pt"


# =========================
# Output folder
# =========================
OUTPUT_PROJECT = BASE_DIR
OUTPUT_NAME = "predict"


def main():

    # ask user for image path
    source_image = input("Enter image path: ").strip()

    source_path = Path(source_image)

    # check image exists
    if not source_path.exists():
        print("Error: Image file not found.")
        return

    # check model exists
    if not MODEL_PATH.exists():
        print("Error: Model file not found.")
        print("Expected location:", MODEL_PATH)
        return

    print("\nLoading model...")
    model = YOLO(str(MODEL_PATH))

    print("Running detection...\n")

    # run prediction
    results = model.predict(
        source=str(source_path),
        save=True,
        project=str(OUTPUT_PROJECT),
        name=OUTPUT_NAME,
        exist_ok=True
    )

    print("\nDetection finished.")
    print("Input image:", source_path)
    print("Output saved to:", OUTPUT_PROJECT / OUTPUT_NAME)

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