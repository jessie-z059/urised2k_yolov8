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

import cv2
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from collections import Counter
import os


BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models/best.pt"
XML_DIR = BASE_DIR / "test_annoed"

CLASS_MAP = {
    'b': 'WBC', 'h': 'RBC', 'pb': 'Broken WBC', 
    'ph': 'Broken RBC', 'm': 'Mycete', 'j': 'Crystal', 's': 'Epithelial'
}

OUTPUT_PROJECT = BASE_DIR
OUTPUT_NAME = "predict"


def main():

    # gnd truth -> pred
    UNIFY = {
        'broken_leukocyte': 'Broken WBC',
        'broken_erythrocyte': 'Broken RBC',
        'leukocyte': 'WBC',
        'erythrocyte': 'RBC',
        'b': 'WBC', 
        'h': 'RBC',
        'pb': 'Broken WBC',
        'ph': 'Broken RBC',
        'sc': 'epithelial',
        's': 'epithelial'
    }

    source_image = input("Enter image path (e.g., test/test.jpg): ").strip()
    source_path = Path(source_image)

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
    pred_counts = Counter()
    for r in results:
        print("\nImage:", r.path)

        if r.boxes is None:
            
            print("No objects detected.")
            continue
        
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            raw_name = r.names[cls_id]
            label_name = UNIFY.get(raw_name, raw_name)
            pred_counts[label_name] += 1
            
            confidence = float(box.conf[0])
            print(f"Detected: {names[cls_id]} | confidence={confidence:.3f}")


    xml_path = XML_DIR / (source_path.stem + ".xml")
    gt_counts = Counter()
    gt_boxes = []

    if xml_path.exists():
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            xml_label = obj.find('name').text
            friendly_name = CLASS_MAP.get(xml_label, xml_label)
            clean_name = UNIFY.get(friendly_name, friendly_name)
            gt_counts[clean_name] += 1
            
            # Store coordinates for plotting
            bbox = obj.find('bndbox')
            gt_boxes.append({
                "name": friendly_name,
                "bbox": [int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                         int(bbox.find('xmax').text), int(bbox.find('ymax').text)]
            })
    else:
        print(f"Warning: XML annotation not found at {xml_path}")

    # Create Comparison Table String
    all_categories = sorted(list(set(gt_counts.keys()) | set(pred_counts.keys())))
    table_header = f"{'Category':<15} | {'GT':<5} | {'Pred':<5}"
    table_divider = "-" * 30
    comparison_lines = [table_header, table_divider]
    
    for cat in all_categories:
        comparison_lines.append(f"{cat:<15} | {gt_counts[cat]:<5} | {pred_counts[cat]:<5}")
    
    comparison_text = "\n".join(comparison_lines)
    print("\n" + comparison_text) # Also print to terminal

    # 4. Visualization (XML Plot)
    img = cv2.imread(str(source_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for item in gt_boxes:
        xmin, ymin, xmax, ymax = item['bbox']
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(img, item['name'], (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    plt.figure(figsize=(12, 10))
    plt.imshow(img)
    # Display the comparison table as the title
    plt.title(f"Ground Truth Visualization (Green Boxes)\n\n{comparison_text}", 
             loc='left', y=1.05, fontdict={'family': 'monospace', 'size': 10, 'weight': 'bold'})
    plt.axis('off')
    plt.subplots_adjust(top=0.75)
    plt.show()

if __name__ == "__main__":
    main()


#     def show_annotated_image(img_p, xml_p):
#         img = cv2.imread(img_p)
#         tree = ET.parse(xml_p)
#         root = tree.getroot()

#         for obj in root.findall('object'):
#             label_code = obj.find('name').text
#             name = class_map.get(label_code, label_code)
            
#             # Get coordinates
#             bbox = obj.find('bndbox')
#             xmin = int(bbox.find('xmin').text)
#             ymin = int(bbox.find('ymin').text)
#             xmax = int(bbox.find('xmax').text)
#             ymax = int(bbox.find('ymax').text)

#             # Draw the box and text
#             cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             cv2.putText(img, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Convert BGR to RGB for Matplotlib
#         plt.figure(figsize=(10, 8))
#         plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         plt.axis('off')
#         plt.show()

#     show_annotated_image(image_path, xml_path)



# if __name__ == "__main__":
#     main()