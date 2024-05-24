import os
import copy
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset, input_size, known_percent=0):

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(int(input_size / 0.875), antialias=True),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        splitRoot = os.path.dirname(__file__) + "/splitFiles"
        with open(splitRoot + f"/{dataset}_train.txt", 'r') as f:
            content_train = f.read()
            train_dict = eval(content_train)
        with open(splitRoot + f"/{dataset}_test.txt", 'r') as f:
            content_test = f.read()
            test_dict = eval(content_test)
        self.trainX, self.trainY = list(train_dict['image_path']), np.array(list(train_dict['label']))
        self.testX, self.testY = list(test_dict['image_path']), np.array(list(test_dict['label']))

        self.classnum_all = len(set(self.testY)) # all classes
        self.classnum_known = int(self.classnum_all * known_percent) # known classes

        self.total_classes = np.arange(self.classnum_all)
        np.random.shuffle(self.total_classes)
        self.known_class_list = sorted(self.total_classes[:self.classnum_known])
        self.unknown_class_list = sorted(self.total_classes[self.classnum_known:])
        # print(" ==> Known class list:\n", self.known_class_list)
        # print(" ==> Unknown class list:\n", self.unknown_class_list)

        # relabel mapping
        self.knowndict = {value: idx for idx, value in enumerate(self.known_class_list)}
        known_len = len(self.knowndict)
        self.unknowndict = {value: known_len + idx for idx, value in enumerate(self.unknown_class_list)}
        # print(" ==> Known dict:\n", self.knowndict)
        # print(" ==> Unknown dict:\n", self.unknowndict)

        self.trainData, self.testData = self.path2data()
        assert(len(self.trainData) == len(self.trainY))
        assert(len(self.testData) == len(self.testY))
 
        self.copytrainY = copy.deepcopy(self.trainY)
        self.copytestY = copy.deepcopy(self.testY)
        for i in range(len(self.known_class_list)):
            self.trainY[self.copytrainY == self.known_class_list[i]] = self.knowndict[self.known_class_list[i]]
            self.testY[self.copytestY == self.known_class_list[i]] = self.knowndict[self.known_class_list[i]]
        for j in range(len(self.unknown_class_list)):
            self.trainY[self.copytrainY == self.unknown_class_list[j]] = self.unknowndict[self.unknown_class_list[j]]
            self.testY[self.copytestY == self.unknown_class_list[j]] = self.unknowndict[self.unknown_class_list[j]]

        self.origin_known_list = self.known_class_list
        self.origin_unknown_list = self.unknown_class_list
        self.new_known_list = np.arange(known_len)
        self.new_unknown_list = np.arange(known_len, known_len + len( self.unknown_class_list))

        self.trian_data_known_index = []
        self.test_data_known_index = []
        for item in self.new_known_list:
            train_index = list(np.where(self.trainY == item)[0])
            self.trian_data_known_index = self.trian_data_known_index + train_index
            test_index = list(np.where(self.testY == item)[0])
            self.test_data_known_index = self.test_data_known_index + test_index

        self.train_data_index_perm = np.arange(len(self.trainY))
        self.train_data_unknown_index = np.setdiff1d(self.train_data_index_perm, self.trian_data_known_index)
        self.test_data_index_perm = np.arange(len(self.testY))
        self.test_data_unknown_index = np.setdiff1d(self.test_data_index_perm, self.test_data_known_index)
        assert (len(self.test_data_unknown_index) + len(self.test_data_known_index) == len(self.testY))

        self.trainX, self.trainY = torch.tensor(self.trainData), torch.tensor(self.trainY)
        self.testX, self.testY= torch.tensor(self.testData), torch.tensor(self.testY)

        self.traindataX = (self.trainX[self.trian_data_known_index]).float()
        self.traindataY = (self.trainY[self.trian_data_known_index]).long()
        self.testclosedataX = (self.testX[self.test_data_known_index]).float()
        self.testclosedataY = (self.testY[self.test_data_known_index]).long()
        self.testopendataX = (self.testX[self.test_data_unknown_index]).float()
        self.testopendataY = (self.testY[self.test_data_unknown_index]).long()

    def known_class_show(self):
        return self.origin_known_list, self.origin_unknown_list
    
    def path2data(self):
        train_data, test_data = [], []
        for path in self.trainX:
            img = Image.open((path)).convert('RGB')
            img = self.transform_train(img).unsqueeze(0)
            train_data.append(img)
        for path in self.testX:
            img = Image.open((path)).convert('RGB')
            img = self.transform_test(img).unsqueeze(0)
            test_data.append(img)
        train_data = np.concatenate(train_data, axis=0)
        test_data = np.concatenate(test_data, axis=0)
        return train_data, test_data
