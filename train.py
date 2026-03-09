from ultralytics import YOLO
import os


def main():

    model = YOLO("yolov8n.pt")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    model.train(
        data="dataset.yaml",
        epochs=1,
        imgsz=10,
        batch=16,
        project=os.path.join(base_dir, "runs"),
        name="train"
    )


if __name__ == "__main__":
    main()