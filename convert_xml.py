import os
import xml.etree.ElementTree as ET

xml_dir = "../UriSed2K/Annotations"
img_dir = "../UriSed2K/iamges"

label_dir = "dataset/labels/train"
img_out = "dataset/images/train"

os.makedirs(label_dir, exist_ok=True)
os.makedirs(img_out, exist_ok=True)

classes = [
"b",
"h",
"pb",
"ph",
"m",
"j",
"s",
"sc",
"ec"
]

for xml_file in os.listdir(xml_dir):

    tree = ET.parse(os.path.join(xml_dir, xml_file))
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    txt_name = xml_file.replace(".xml", ".txt")

    with open(os.path.join(label_dir, txt_name), "w") as f:

        for obj in root.iter("object"):

            cls = obj.find("name").text
            cls_id = classes.index(cls)

            xmlbox = obj.find("bndbox")

            xmin = float(xmlbox.find("xmin").text)
            xmax = float(xmlbox.find("xmax").text)
            ymin = float(xmlbox.find("ymin").text)
            ymax = float(xmlbox.find("ymax").text)

            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h

            f.write(f"{cls_id} {x_center} {y_center} {width} {height}\n")

    # copy image
    img_name = xml_file.replace(".xml", ".jpg")
    src = os.path.join(img_dir, img_name)
    dst = os.path.join(img_out, img_name)

    if os.path.exists(src):
        os.system(f"cp '{src}' '{dst}'")