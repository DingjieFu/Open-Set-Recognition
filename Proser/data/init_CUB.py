#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File : CUB_split.py
# @Author : DingjieFu
# @Time : 2024/05/02 09:07:02
"""
    - split CUB
"""
import os
import numpy as np



imageRoot = "dataRoot/FGVC"
origin_path = imageRoot + "/CUB/CUB_200_2011"

# get image paths
with open(os.path.join(origin_path, "images.txt")) as f:
    lines = f.readlines()
    paths_suffix = [line.strip().split()[-1] for line in lines]
    path_all = [origin_path + "/images/" + path_suffix for path_suffix in paths_suffix]

# get corresponding labels
with open(os.path.join(origin_path, "image_class_labels.txt")) as f:
    lines = f.readlines()
    label_all = np.array([int(line.strip().split()[-1]) for line in lines])

# get classnames
with open(os.path.join(origin_path, "classes.txt")) as f:
    lines = f.readlines()
    class_names = [line.strip().split()[-1] for line in lines]

# split data
with open(os.path.join(origin_path, "train_test_split.txt")) as f:
    lines = f.readlines()
    image_idx = [int(line.strip().split()[0]) for line in lines]
    is_train = np.array([int(line.strip().split()[1]) for line in lines])

train_idxes = np.where(is_train == 1)[0]
test_idxes = np.where(is_train == 0)[0]

# save splitted data
path_train = [path_all[idx] for idx in train_idxes]
label_train = label_all[train_idxes] - 1

path_test = [path_all[idx] for idx in test_idxes]
label_test = label_all[test_idxes] - 1

test_label2path, test_label2imgid = [], []
for label in np.unique(label_test):
    label_idxes = np.where(label_test == label)[0]
    test_label2path.append([path_test[idx] for idx in label_idxes])
    test_label2imgid.append([test_idxes[idx] for idx in label_idxes])


CUB_dict_train = {"image_path": path_train, "class_names": class_names, 
                  "label": label_train.tolist(), "indexes": train_idxes.tolist()}
CUB_dict_test = {'image_path': path_test, 'class_names': class_names, 
                 'label': label_test.tolist(), 'indexes': test_idxes.tolist()}
CUB_dict = {'class_path': test_label2path, 'class_name': class_names, 'class_image': test_label2imgid}


filePath = os.path.abspath(__file__)
now_dir = os.path.dirname(filePath)
split_dir = now_dir + "/splitFiles"
os.makedirs(split_dir, exist_ok=True)

with open(os.path.join(split_dir, 'CUB_train.txt'), 'w') as f:
    f.write(str(CUB_dict_train))

with open(os.path.join(split_dir, 'CUB_test.txt'), 'w') as f:
    f.write(str(CUB_dict_test))

with open(os.path.join(split_dir, 'CUB_dummy.txt'), 'w') as f:
    f.write(str(CUB_dict))

print("Finish!")
