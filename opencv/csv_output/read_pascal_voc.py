import os
import xml.etree.ElementTree as ET


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    obj_names = []

    for boxes in root.iter('object'):

        names = boxes.findall('name')
        for name in names:
            obj_names.append(name.text)

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return obj_names, list_with_all_boxes

names, boxes = read_content("/home/duclv/homework/dataset/katech/신성례/새 폴더/20191018_161518(송파역_잠실역 평지 교량구간_차로변경 및 선행차량 다차로 차선변경)_261f/3_annotations_v001_1/3_20191018_161518_000000_v001_1.xml")
print(names)
print(boxes)
