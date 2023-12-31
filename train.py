import os

DEVICE_ID = 5
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import json
import shutil
import time
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



FOLDS_ROOT = "loo_stained_wsi/"
# FOLDS_ROOT = '/kaggle/input/hubmap-folds/

PREFIX = "loo_stained"
PROJECT = "HuBMAP"
MODEL_V = "yolov8x-seg"

folds_i = [2, 3]
# folds_i = [3, 4]


for i in folds_i:
    path_to_data = f"{FOLDS_ROOT}/fold{i}/hubmap-coco.yaml"

    model = YOLO(MODEL_V)
    model.train(
        task='segment',
        project=PROJECT,
        name=f"{PREFIX}_{MODEL_V}-fold{i}",
        # Random Seed parameters
        deterministic=True,
        seed=42,
        # Training parameters
        data=path_to_data,
        single_cls=False,
        save=True,
        save_period=10,
        pretrained=True,
        # pretrained=f"{PROJECT}/only_ds2_yolov8x-seg-fold{i}/weights/best.pt",
        imgsz=512,
        epochs=200,
        batch=100,
        workers=3,
        val=True,
        fraction=0.8,
        device=DEVICE_ID,
        dfl=3,  # 1.5 default, maybe 3 better
        box=5,  # 7.5 default
        retina_masks=True,

        # Optimization parameters
        lr0=3e-4,
        patience=10,
        cos_lr=True,
        optimizer="AdamW",
        weight_decay=0.001,

        # Augmentation parameters
        overlap_mask=False,
        augment=True,
        mosaic=0.,
        mixup=0.,
        degrees=90.0,
        translate=0.3,
        scale=0.5,
        shear=15.0,
        perspective=0.0005,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.04,  # image HSV-Hue augmentation (fraction) default 0.015
        hsv_s=0.4,  # (float) image HSV-Saturation augmentation (fraction) default 0.7
        hsv_v=0.3,  # (float) image HSV-Value augmentation (fraction) default 0.4
        # copy_paste=0.2, # strong augmentation
    )
    del model

    time.sleep(20)
