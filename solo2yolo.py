# Convert solo dataset to yolov8 format
# This is specialized to the output from the Unity Perception package.
# Using datasets from other tools may require modifications to this script.

# file deepcode ignore : your mother

import os
import sys
import argparse
import json
import yaml
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

argparser = argparse.ArgumentParser(description='Convert solo dataset to yolov8 format')
argparser.add_argument('-i', '--input', help='path to solo dataset', required=True)
argparser.add_argument('-o', '--output', help='path to yolov8 dataset', required=True)
argparser.add_argument('-ttv', '--train_test_val', help='train/test/val split (must add up to 100%)', required=False, default='70/20/10')
argparser.add_argument('-r', '--random', help='randomize sequence order', required=False, default=False, action='store_true')

args = argparser.parse_args(sys.argv[1:])
solo_path = os.path.abspath(args.input)
yolov8_path = args.output
percent_train, percent_test, percent_val = [int(i) for i in args.train_test_val.split('/')]

data_yml = {}

if percent_train + percent_test + percent_val != 100:
    print('Error: train/test/val split must add up to 100%')
    sys.exit(1)

if not os.path.exists(solo_path):
    print('Error: Solo dataset not found')
    sys.exit(1)

if not os.path.exists(yolov8_path):
    os.makedirs(yolov8_path)
else:
    if os.listdir(yolov8_path) != []:
        i = input(f"Warning: yolov8 dataset directory {yolov8_path} is not empty. Continue? (y/n) ")
        if i.lower() != 'y':
            sys.exit(1)
        
        os.system(f"rm -rf {yolov8_path}/*")

sequence_list = []
for sequence in os.listdir(solo_path):
    if sequence.startswith('sequence.') and not sequence.endswith('.0'): # sequence.0 (at least from unity perception) is garbage
        sequence_list.append(sequence)
print(f"Found {len(sequence_list)} sequences")

class_list = []
with open(os.path.join(solo_path, 'annotation_definitions.json'), 'r') as f:
    print(f"Loading annotation definitions from {os.path.join(yolov8_path, 'annotation_definitions.json')}")
    data = json.load(f)
    for class_info in data['annotationDefinitions'][0]['spec']:
        class_list.append(class_info['label_name'])

data_yml['names'] = class_list
data_yml['nc'] = len(class_list)

print(f"Found the following ({len(class_list)}) classes:")
for class_name in class_list:
    print(f" - {class_name}")

print(f"Splitting into {percent_train}% train, {percent_test}% test, {percent_val}% val")
num_train = int(len(sequence_list) * int(percent_train) / 100)
num_test = int(len(sequence_list) * int(percent_test) / 100)
num_val = int(len(sequence_list) * int(percent_val) / 100)
print(f"Train: {num_train}; Test: {num_test}; Val: {num_val}")
train_images = []
test_images = []
val_images = []

if args.random:
    import random
    random.shuffle(sequence_list)

for dir in ['train', 'test', 'val']:
    if not os.path.exists(os.path.join(yolov8_path, dir)):
        os.makedirs(os.path.join(yolov8_path, dir))
    if not os.path.exists(os.path.join(yolov8_path, dir, 'images')):
        os.makedirs(os.path.join(yolov8_path, dir, 'images'))
    if not os.path.exists(os.path.join(yolov8_path, dir, 'labels')):
        os.makedirs(os.path.join(yolov8_path, dir, 'labels'))

data_yml['train'] = "train/images"
data_yml['test'] = "test/images"
data_yml['val'] = "val/images"

print('Writing data.yaml')
with open(os.path.join(yolov8_path, 'data.yaml'), 'w') as f:
    yaml.dump(data_yml, f)

train_index, test_index, val_index = 0, 0, 0

labels = {i: {'labels': [], 'version': 'synthetic'} for i in ['train', 'test', 'val']}

for sequence in tqdm(sequence_list, desc='Writing images and labels', ncols=100, unit=' writes'): # get a nice progress bar
    if not os.path.isdir(os.path.join(solo_path, sequence)):
        continue
    if train_index < num_train:
        ttv = 'train'
        train_index += 1
    elif test_index < num_test:
        ttv = 'test'
        test_index += 1
    elif val_index < num_val:
        ttv = 'val'
        val_index += 1
    image_path = os.path.join(solo_path, sequence, 'step0.camera.png')
    data_path = os.path.join(solo_path, sequence, 'step0.frame_data.json')
    annotations_txt = ""
    with open(data_path, 'r') as f:
        data = json.load(f)
        for capture in data['captures']:
            for annotation in capture['annotations']:
                if annotation['id'] == 'bounding box':
                    classes = []
                    bboxes = []
                    y, x = np.asarray(Image.open(image_path)).shape[:2] # marginally faster than cv2.imread. also, for some reason, .shape returns y, x
                    for value in annotation['values']:
                        origin = value['origin']
                        dimension = value['dimension']
                        # normalize to 0.0-1.0
                        box_center_x = (origin[0] + dimension[0] / 2) / x
                        box_center_y = (origin[1] + dimension[1] / 2) / y
                        box_width = dimension[0] / x
                        box_height = dimension[1] / y
                        bbox = [box_center_x, box_center_y, box_width, box_height]
                        for i in bbox:
                            if i < 0.0 or i > 1.0:
                                raise Exception(f"Error: bbox {bbox} is out of bounds at {bbox.index(i)}\nYour dataset is likely corrupted.")
                        annotations_txt += f"{class_list.index(value['labelName'])} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                        classes.append(class_list.index(value['labelName']))
                        bboxes.append(bbox)

                    data = {
                        'im_file': os.path.join(yolov8_path, ttv, 'images', f"{sequence}.png"),
                        'shape': (x, y),
                        'cls': np.asarray(set(classes)),
                        'bboxes': np.array(bboxes, dtype=np.float32),
                        'segments': [],
                        'keypoints': None,
                        'normalized': True,
                        'bbox_format': 'xywh'
                    }
                    labels[ttv]['labels'].append(data)
                    classes = []
    
    shutil.copyfile(image_path, os.path.join(yolov8_path, ttv, 'images', f"{sequence}.png"))
    with open(os.path.join(yolov8_path, ttv, 'labels', f"{sequence}.txt"), 'w') as f:
        f.write(annotations_txt)

print('Writing labels.cache')
for i in labels.keys():
    np.save(os.path.join(yolov8_path, i, 'labels.cache'), labels[i])