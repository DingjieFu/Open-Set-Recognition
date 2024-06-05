#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File : init_StanfordCars.py
# @Author : DingjieFu
# @Time : 2024/05/29 21:25:23
"""
    -split StanfordCars
"""
import os
import numpy as np
import scipy.io as sio



imageRoot = "/mimer/NOBACKUP/groups/alvis_cvl/Fahad/vision_tasks_L/ybxmw/dataRoot/FGVC"
origin_path = os.path.join(imageRoot,'Stanford_Car')

matFile = os.path.join(origin_path,'cars_annos.mat')
dataMat = sio.loadmat(matFile)['annotations'][0]

path_all = [origin_path + "/" + matData[0][0] for matData in dataMat]
label_all = np.array([int(matData[5][0][0]) - 1 for matData in dataMat])
is_train = np.array([int(matData[6][0][0]) for matData in dataMat])
assert(len(path_all) == len(label_all))

train_idxes = np.where(is_train == 1)[0]
test_idxes = np.where(is_train == 0)[0]

# save splitted data
path_train = [path_all[idx] for idx in train_idxes]
label_train = label_all[train_idxes]

path_test = [path_all[idx] for idx in test_idxes]
label_test = label_all[test_idxes]

test_label2path, test_label2imgid = [], []
for label in np.unique(label_test):
    label_idxes = np.where(label_test == label)[0]
    test_label2path.append([path_test[idx] for idx in label_idxes])
    test_label2imgid.append([test_idxes[idx] for idx in label_idxes])


SCars_dict_train = {"image_path": path_train, "label": label_train.tolist()}
SCars_dict_test = {'image_path': path_test, 'label': label_test.tolist()}
SCars_dict = {'class_path': test_label2path, 'class_image': test_label2imgid}


filePath = os.path.abspath(__file__)
now_dir = os.path.dirname(filePath)
split_dir = now_dir + "/splitFiles"
os.makedirs(split_dir, exist_ok=True)

with open(os.path.join(split_dir, 'Stanford_Cars_train.txt'), 'w') as f:
    f.write(str(SCars_dict_train))

with open(os.path.join(split_dir, 'Stanford_Cars_test.txt'), 'w') as f:
    f.write(str(SCars_dict_test))

with open(os.path.join(split_dir, 'Stanford_Cars_dummy.txt'), 'w') as f:
    f.write(str(SCars_dict))

print("Finish!")