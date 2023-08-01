# Create the k-fold directories

import itertools
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import staintools
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

DATA_DIR = Path('../data/')

# Read .jsonl file and convert it to a list of dicts
# The dicts contain IDs, class names and segmentation masks
# from https://www.kaggle.com/code/leonidkulyk/eda-hubmap-hhv-interactive-annotations
with open(DATA_DIR / 'cleaned_polygons.jsonl', 'r') as json_file:
    json_labels = [json.loads(line) for line in json_file]

id_to_annotation = {j['id']: j['annotations'] for j in json_labels}

# Define a conversion between class name and number
id_dict = {'blood_vessel': 0, 'glomerulus': 1, 'unsure': 2}

# Function to copy images and transform labels to 
# coco formatted .txt files
def tile_to_coco(tile_id, annotations, output_folder: Path):
    # Copy image
    shutil.copyfile(DATA_DIR / f'train/{tile_id}.tif', output_folder / f'{tile_id}.tif')
    
    # Create text file and write formatted labels to it
    with open(output_folder / f'{tile_id}.txt', 'w') as text_file:
        for annotation in annotations:
            
            class_id = id_dict[annotation['type']]
            if class_id == 2:
                continue
            flat_mask_polygon = list(itertools.chain(*annotation['coordinates'][0]))
            # Divide by 512 because coco labels expect positions between 0 and 1
            # not pixel indices
            array = np.array(flat_mask_polygon)/512.
            text_file.write(f'{class_id} {" ".join(map(str, array))}\n')
            
        

# Function to copy images and transform labels to 
# coco formatted .txt files
def tile_to_coco_stain(tile_id, annotations, output_folder: Path, stain_num=0):
    
    # Read data
    image = staintools.read_image(str(DATA_DIR / f'train/{tile_id}.tif'))
    image = staintools.LuminosityStandardizer.standardize(image)
    
    if stain_num == 0:
        cv2.imwrite(str(output_folder / f'{tile_id}.tif'), image[...,::-1])
        # Create text file and write formatted labels to it
        with open(output_folder / f'{tile_id}.txt', 'w') as text_file:
            for annotation in annotations:

                class_id = id_dict[annotation['type']]
                if class_id == 2:
                    continue
                flat_mask_polygon = list(itertools.chain(*annotation['coordinates'][0]))
                # Divide by 512 because coco labels expect positions between 0 and 1
                # not pixel indices
                array = np.array(flat_mask_polygon)/512.
                text_file.write(f'{class_id} {" ".join(map(str, array))}\n')

    else:
        augmentor = staintools.StainAugmentor(method='vahadane', sigma1=0.3, sigma2=0.3)
        augmentor.fit(image)
        for i in range(stain_num+1):
            if i != 0:
                image = augmentor.pop().astype(np.uint8)
                
            cv2.imwrite(str(output_folder / f'{tile_id}_{i}.tif'), image[...,::-1])
            # Create text file and write formatted labels to it
            with open(output_folder / f'{tile_id}_{i}.txt', 'w') as text_file:
                for annotation in annotations:

                    class_id = id_dict[annotation['type']]
                    if class_id == 2:
                        continue
                    flat_mask_polygon = list(itertools.chain(*annotation['coordinates'][0]))
                    # Divide by 512 because coco labels expect positions between 0 and 1
                    # not pixel indices
                    array = np.array(flat_mask_polygon)/512.
                    text_file.write(f'{class_id} {" ".join(map(str, array))}\n')

meta = pd.read_csv(DATA_DIR / 'tile_meta.csv')

meta.dataset.value_counts()

rows = []
with open(f'{DATA_DIR}/cleaned_polygons.jsonl', 'r') as json_file:
    for line in json_file:
        data = json.loads(line)
        row = dict({'id': data['id']})
        coords = []
        for ann in data['annotations']:
            if ann['type'] == 'blood_vessel':
                coords.append(ann['coordinates'])
        row['annotation'] = coords
        
        rows.append(row)
df = pd.DataFrame(rows)

df = df.merge(meta, on='id')

# df = df[df.dataset == 1]

df['num_cells'] = df['annotation'].apply(lambda x: len(x))

df["area"] = df["annotation"].apply(lambda xs: sum([cv2.contourArea(np.array(x)) for x in xs])/512**2) # You might need a different function to calculate the area depending on the structure of annotation

import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure with two subplots
fig, axes = plt.subplots(3, figsize=(10, 20))

# Plot histograms of areas
sns.histplot(df, x="area", hue="source_wsi", element="step", stat="density", common_norm=False, ax=axes[0])
axes[0].set_title('Area Distribution by source_wsi')

sns.boxplot(df, x="dataset", y="area", ax=axes[1])
axes[1].set_title('Area Distribution by dataset')

# Plot boxplots of num_cells
sns.boxplot(x="source_wsi", y="num_cells", data=df, ax=axes[2])
axes[2].set_title('Number of Cells by source_wsi')

plt.tight_layout()
plt.show()

df["num_cells_binned"] = pd.qcut(df["num_cells"], q=4, labels=False, duplicates='drop')
df["area_binned"] = pd.qcut(df["area"], q=4, labels=False, duplicates='drop')

df_encoded = pd.get_dummies(df, columns=["num_cells_binned", "area_binned", "source_wsi"])

stratify_cols = [col for col in df_encoded.columns if "num_cells_binned" in col or "area_binned" in col or "source_wsi" in col]
df_stratify = df_encoded[stratify_cols]

# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import StandardScaler

# # Извлечь данные из dataset 1 и dataset 2
# df1 = df[df.dataset==1]
# df2 = df[df.dataset==2]

# # Выделить признаки для сравнения
# features = ['area', 'num_cells']
# df1_features = df1[features]
# df2_features = df2[features]

# # Масштабировать признаки
# scaler = StandardScaler()
# df1_features_scaled = scaler.fit_transform(df1_features)
# df2_features_scaled = scaler.transform(df2_features)

# # Обучить модель NearestNeighbors на dataset 1
# nbrs = NearestNeighbors(n_neighbors=1).fit(df1_features_scaled)

# # Найти ближайшие соседи в dataset 1 для каждого элемента в dataset 2
# distances, indices = nbrs.kneighbors(df2_features_scaled)

# # Отбор индексов dataset 2, которые ближе всего к dataset 1
# df2_closest = df2.iloc[indices.flatten()]

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=5)

## Folds directories

FOLD_DIR = 'loo_stained_wsi'

# Assuming k=5 for 5-fold cross-validation
# k = 5
k = 4

# Create the k-fold directories
for i in range(k):
    fold_dir = f'{FOLD_DIR}/fold{i}/'
    os.makedirs(fold_dir)
    os.makedirs(fold_dir + 'train/')
    os.makedirs(fold_dir + 'valid/')

## Split Data

num_dict = {1: 2, 2: 2, 3: 2, 3: 2}

# from tqdm.notebook import tqdm_notebook as tqdm
from tqdm import tqdm

for fold_index, source_wsi in enumerate(sorted(df.source_wsi.unique())):
    valid_idx = df[df.source_wsi == source_wsi]['id'].values
    train_idx = df[df.source_wsi != source_wsi]['id'].values
    
    fold_dir = f'{FOLD_DIR}/fold{fold_index}/'
    
    # Copy the train and valid data to the corresponding fold directory
    for i in tqdm(train_idx):
        tile_to_coco_stain(i, id_to_annotation[i] , Path(fold_dir + 'train/'), stain_num=5)
    for j in valid_idx:
        tile_to_coco_stain(j, id_to_annotation[j] , Path(fold_dir + 'valid/'), stain_num=0)

    # Create the hubmap-coco.yaml file for each fold
    yaml_text = f"""
    # HuBMAP - Hacking the Human Vasculature dataset 
    # https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature

    # train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
    train: {str(Path(fold_dir + 'train/').absolute())}
    val: {str(Path(fold_dir + 'valid/').absolute())}

    # class names
    names: 
      0: blood_vessel
      1: glomerulus
      
    """

    with open(fold_dir + 'hubmap-coco.yaml', 'w') as text_file:
        text_file.write(yaml_text)
