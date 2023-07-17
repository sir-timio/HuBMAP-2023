import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import json
import shutil
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
import torch
import ultralytics
import yaml
from PIL import Image
from pycocotools import _mask as coco_mask
from tqdm.notebook import tqdm
from ultralytics import YOLO

BASE_FILENAME = "folds/"
# BASE_FILENAME = '/kaggle/input/hubmap-hacking-the-human-vasculature'

ultralytics.checks()


def parse_ann(annotations):
    target = []
    for annot in annotations:
        if annot["type"] == "blood_vessel":
            target.append(np.array(annot["coordinates"]).flatten().tolist())
    return target


from dataclasses import dataclass


@dataclass
class CFG:
    dilation_n_iter: int = 1
    conf: float = 0.01
    imgsz: int = (512, 512)
    retina_masks: bool = True
    iou_nms: float = 0.3

    author: str = "tg @ai_minds"
    yolo_path: str = "/kaggle/input/models-hubmap-vasculative/yolov8x-seg1.pt"

    def __repr__(self):
        return f"CFG({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"


# Metrics

from metrics import MAPCalculatorSingleClass


def predict_yolo(model, image, imgsz=1024, conf=0.01, iou_nms=0.5, retina_masks=True):
    classes = model.names  # {0: 'blood_vessel', 1: 'glomerulus', 2: 'unsure'}
    result = dict()
    pred = model.predict(
        image, imgsz=imgsz, conf=conf, iou=iou_nms, retina_masks=retina_masks
    )[0]
    orig_shape = pred.orig_shape

    for k in classes.keys():
        result[k] = dict(
            {"masks": np.zeros(shape=(1, *orig_shape)), "confs": np.array([0])}
        )

    if pred.masks is None:
        return result

    for k in classes.keys():
        masks = pred.masks.data[pred.boxes.cls == k].detach().cpu().numpy()
        if len(masks) == 0:
            continue
        confs = pred.boxes.conf[pred.boxes.cls == k].detach().cpu().numpy()
        if imgsz != orig_shape:
            scaled_masks = []
            for mask in masks:
                scaled_mask = scale_image(mask, orig_shape)
                if len(scaled_mask.shape) == 3:
                    scaled_mask = scaled_mask[:, :, 0]
                scaled_masks.append(scaled_mask)
            masks = np.stack(scaled_masks)
        result[k] = dict({"masks": masks, "confs": confs})

    return result


# Training

with open("../data/polygons.jsonl", "r") as json_file:
    json_labels = [json.loads(line) for line in json_file]

ANN_DICT = {j["id"]: j["annotations"] for j in json_labels}

DATA_BASE = "dataset"
BASE_FILENAME = "../data/"


def evaluate_model(annotations, predictions):
    mAP_calc = MAPCalculatorSingleClass()
    height, width = (512, 512)
    ious = []
    for ann, pred in zip(annotations, predictions):
        if len(ann) == 0:
            continue
        enc_gt = coco_mask.frPyObjects(ann, height, width)
        num_gts = len(enc_gt)
        pred_masks = pred[0]["masks"].astype(np.uint8)
        scores = pred[0]["confs"]

        enc_pred = [mask_util.encode(np.asarray(p, order="F")) for p in pred_masks]

        _ious = mask_util.iou(enc_pred, enc_gt, [0] * len(enc_gt))
        mAP_calc.accumulate(_ious, scores, num_gts)
        score = mAP_calc.evaluate()[0]
        ious.append(score)
    return ious


FOLDS_ROOT = "folds/"
# FOLDS_ROOT = '/kaggle/input/hubmap-folds/

PROJECT = "HuBMAP"

MODEL_V = "yolov8x-seg"

# folds_i = [0, 1, 2, 3, 4]
folds_i = [2, 3, 4]


for i in folds_i:
    #     _path_to_data = f'{FOLDS_ROOT}/fold{i}/fold{i}/hubmap-coco.yaml'
    #     config = yaml.load(open(_path_to_data), Loader=yaml.BaseLoader)
    #     config['train'] = f'/kaggle/input/hubmap-folds/fold{i}/fold{i}/train'
    #     config['val'] = f'/kaggle/input/hubmap-folds/fold{i}/fold{i}/valid'

    #     os.makedirs('/kaggle/working/folds_configs', exist_ok=1)
    #     path_to_data = f'/kaggle/working/folds_configs/fold{i}.yaml'
    #     with open(path_to_data, 'w') as file:
    #         yaml.dump(config, file)
    path_to_data = f"{FOLDS_ROOT}/fold{i}/hubmap-coco.yaml"

    model = YOLO(MODEL_V)
    model.train(
        project=PROJECT,
        name=f"{MODEL_V}-fold{i}",
        # Random Seed parameters
        deterministic=True,
        seed=42,
        # Training parameters
        data=path_to_data,
        save=True,
        save_period=5,
        pretrained=True,
        imgsz=512,
        epochs=120,
        batch=64,
        workers=4,
        val=True,
        device=6,
        dfl=3,  # 1.5 default
        box=3,  # 7.5 default
        mask_ratio=1,  # 1 to 1 size
        # Optimization parameters
        lr0=3e-4,
        patience=20,
        optimizer="AdamW",
        weight_decay=0.0005,
        # Augmentation parameters
        overlap_mask=False,
        augment=True,
        mosaic=0.5,
        mixup=0.3,
        degrees=10.0,
        translate=0.2,
        scale=0.5,
        shear=2.0,
        perspective=0,
        flipud=0.5,
        fliplr=0.5,
    )
