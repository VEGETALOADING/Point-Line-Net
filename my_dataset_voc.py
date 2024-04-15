import os
import json

from lxml import etree
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from train_utils import convert_to_coco_api


class VOCInstances(Dataset):
    def __init__(self, voc_root, year="2012", txt_name: str = "train.txt", transforms=None, mode=None):
        super().__init__()
        if isinstance(year, int):
            year = str(year)
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        if "VOCdevkit" in voc_root:
            root = os.path.join(voc_root, f"VOC{year}")
        else:
            root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        xml_dir = os.path.join(root, 'Annotations')

        txt_path = os.path.join(root, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        json_file = 'my_voc_indices.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            idx2classes = json.load(f)
            self.class_dict = dict([(v, k) for k, v in idx2classes.items()])

        self.images_path = []
        self.xmls_path = []
        self.xmls_info = []
        self.objects = []
        self.mode = mode

        images_path = [os.path.join(image_dir, x + ".JPG") for x in file_names]
        xmls_path = [os.path.join(xml_dir, x + '.xml') for x in file_names]
        for idx, (img_path, xml_path) in enumerate(zip(images_path, xmls_path)):
            assert os.path.exists(img_path), f"not find {img_path}"
            assert os.path.exists(xml_path), f"not find {xml_path}"

            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            obs_dict = parse_xml_to_dict(xml)["annotation"]
            obs_bboxes = parse_objects(obs_dict, xml_path, self.class_dict, idx, mode=self.mode)
            num_objs = obs_bboxes["boxes"].shape[0]

            num_instances = obs_bboxes["keypoints"].shape[0]
            if num_objs != num_instances:
                print(f"warning: num_boxes:{num_objs} and num_instances:{num_instances} do not correspond. "
                      f"skip image:{img_path}")
                continue

            self.images_path.append(img_path)
            self.xmls_path.append(xml_path)
            self.xmls_info.append(obs_dict)
            self.objects.append(obs_bboxes)

        self.transforms = transforms
        self.coco = convert_to_coco_api(self)

    def __getitem__(self, idx):

        img = Image.open(self.images_path[idx]).convert('RGB')
        target = self.objects[idx]

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images_path)

    def get_height_and_width(self, idx):

        # read xml
        data = self.xmls_info[idx]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def get_annotations(self, idx):
        data = self.xmls_info[idx]
        h = int(data["size"]["height"])
        w = int(data["size"]["width"])
        target = self.objects[idx]
        return target, h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


def parse_xml_to_dict(xml):


    if len(xml) == 0:
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def parse_objects(data: dict, xml_path: str, class_dict: dict, idx: int, mode: str):

    if mode == 'val':
        boxes = []
        labels = []
        iscrowd = []
        keypoints = []
        obj_idxs = []
        linepoints = []

        assert "object" in data, "{} lack of object information.".format(xml_path)
        obj_idx = 0
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])


            keypoints_xml = kpStr2List(obj["keypoints"])
            linepoints_xml = lpStr2List(obj["linepoints"], maxLength=60)

            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(class_dict[obj["name"]]))
            keypoints.append(keypoints_xml)
            linepoints.append(linepoints_xml)

            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

            obj_idxs.append(obj_idx)
            obj_idx += 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        linepoints = torch.as_tensor(linepoints, dtype=torch.float32)
        image_id = torch.tensor([idx])
        obj_idxs = torch.as_tensor(obj_idxs, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return {
                "boxes": boxes,
                "ori_boxes": boxes,
                "labels": labels,
                "iscrowd": iscrowd,
                "image_id": image_id,
                "area": area,
                "keypoints": keypoints,
                "ori_linepoints": linepoints,
                "obj_idxs": obj_idxs,
                }
    else:
        boxes = []
        labels = []
        iscrowd = []
        keypoints = []
        obj_idxs = []

        assert "object" in data, "{} lack of object information.".format(xml_path)
        obj_idx = 0
        for obj in data["object"]:
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            keypoints_xml = kpStr2List(obj["keypoints"])


            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(class_dict[obj["name"]]))
            keypoints.append(keypoints_xml)

            if "difficult" in obj:
                iscrowd.append(int(obj["difficult"]))
            else:
                iscrowd.append(0)

            obj_idxs.append(obj_idx)
            obj_idx += 1

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
        image_id = torch.tensor([idx])
        obj_idxs = torch.as_tensor(obj_idxs, dtype=torch.int64)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        return {
            "boxes": boxes,
            "labels": labels,
            "iscrowd": iscrowd,
            "image_id": image_id,
            "area": area,
            "keypoints": keypoints,
            "obj_idxs": obj_idxs,
        }
def kpStr2List(str):
    kpList = []
    numList =str[1:-1].split(', ')

    for index in range(0, len(numList), 2):
        kpList.append([int(numList[index]),
                       int(numList[index + 1]),
                       0 if int(numList[index]) == 0 and int(numList[index + 1]) == 0 else 2])

    assert len(kpList) == 10, '处理错误'

    return kpList
def lpStr2List(str, maxLength):
    lpList = []
    pointStrs = str[2:-2].split('], [')

    for pointStr in pointStrs:
        x, y = pointStr.split(', ')
        lpList.append(
            [int(x), int(y)]
        )

    for _ in range(maxLength - len(lpList)):
        lpList.append([0, 0])

    assert len(lpList) == maxLength, '处理错误'

    return lpList
