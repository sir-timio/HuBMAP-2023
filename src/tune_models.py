import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from ultralytics import YOLO

FOLDS_ROOT = "../data/tune_folds/"

PREFIX = "ds1_tune_after_ds2"
PROJECT = "HuBMAP"
MODEL_V = "yolov8x-seg"

folds_i = [0, 1, 2, 3, 4]


for i in folds_i:

    path_to_data = f"{FOLDS_ROOT}/fold{i}.yaml"

    model = YOLO(MODEL_V)
    model.train(
        project=PROJECT,
        name=f"{PREFIX}_{MODEL_V}-fold{i}",
        # resume=f'checkpoints/ds2_{MODEL_V}-fold{i}',
        # Random Seed parameters
        deterministic=True,
        seed=42,
        # Training parameters
        data=path_to_data,
        single_cls=True,
        save=True,
        save_period=10,
        pretrained=f'checkpoints/ds2_{MODEL_V}-fold{i}',
        imgsz=512,
        epochs=250,
        batch=64,
        workers=1,
        val=True,
        device=5,
        dfl=5,  # 1.5 default
        box=3,  # 7.5 default
        mask_ratio=1,  # 1 to 1 size
        retina_masks=True,
        # Optimization parameters
        lr0=3e-5,
        patience=10,
        cos_lr=True,
        optimizer="AdamW",
        weight_decay=0.001,
        # Augmentation parameters
        overlap_mask=False,
        augment=True,
        mosaic=0.5,
        mixup=0.5,
        degrees=90.0,
        translate=0.2,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.5,
        fliplr=0.5,
    )
    del model
