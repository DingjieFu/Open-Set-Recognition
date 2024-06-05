#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# @File : pretrain_unknown_detection.py
# @Author : DingjieFu
# @Time : 2024/05/03 20:41:26
"""
    - train close-set classifier
"""
import os
import time
import random
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import TensorDataset, DataLoader

from config import parse_args
from data.myLoader import MyDataset


def create_unique_folder_name(base_folder_path):
    count = 0
    new_folder_name = base_folder_path
    while os.path.exists(new_folder_name):
        count += 1
        new_folder_name = f"{base_folder_path}({count})"
    return new_folder_name


args = parse_args()
print(vars(args))
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = args.device

# ======================================== Data prepare ======================================== #
print("\n ==> Preparing data")
mydataset = MyDataset(args.dataset, args.input_size, args.known_percent)

traindataX, traindataY = mydataset.traindataX, mydataset.traindataY
trainset = TensorDataset(traindataX, traindataY)
trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

testclosedataX, testclosedataY = mydataset.testclosedataX, mydataset.testclosedataY
closeset = TensorDataset(testclosedataX, testclosedataY)
closeloader = DataLoader(closeset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# ======================================== Load model ======================================== #
print(" ==> Building model..")
if args.backbone == "Resnet50":
    net = models.resnet50(weights='DEFAULT')
    num_ftrs = net.fc.in_features
    net.fc = nn.Sequential(nn.Linear(num_ftrs, mydataset.classnum_known))
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = MultiStepLR(optimizer, milestones=[50, 125], gamma=0.1)

rootDir = os.path.join(args.log_root, args.dataset)
save_path = f"{args.backbone}-Lr{args.lr}-Seed{args.seed}"
args.save_path = os.path.join(rootDir, save_path)
os.makedirs(args.log_root, exist_ok=True)
os.makedirs(rootDir, exist_ok=True)
os.makedirs(args.save_path, exist_ok=True)

# ======================================== Log ======================================== #
outlogPath = os.path.join(args.save_path,  "pretrain")
os.makedirs(outlogPath, exist_ok=True)
t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
log = outlogPath + "/" + t + "_pretrain"  + '.txt'
logging.basicConfig(format='%(message)s', level=logging.INFO, filename=log, filemode='w')
logger = logging.getLogger(__name__)
argsDict = args.__dict__
for eachArg, value in argsDict.items():
    logger.info(eachArg + ':' + str(value))
logger.info("="*50)

# ======================================== Train Close-set Classifier ======================================== #
best_acc = 0.0
early_stop = 0
for epoch in range(0, args.nepoch):
    # ======================================== Train Stage ======================================== #
    print("\n==> Epoch: %d" % epoch)
    logger.info("\n==> Epoch: %d" % epoch)
    
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    progress = tqdm(total=len(trainloader), ncols=100, desc='Train {}'.format(epoch))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        progress.update(1)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    progress.close()
    print("Train Loss: %.3f; Train acc: %.3f%%" % (train_loss/(batch_idx + 1), 100.0 * correct / total))
    logger.info("Train Loss: %.3f; Train acc: %.3f%%" % (train_loss/(batch_idx + 1), 100.0 * correct / total))
    scheduler.step()
    # ======================================== Test Stage ======================================== #
    
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        progress = tqdm(total=len(closeloader), ncols=100, desc='Test {}'.format(epoch))
        for batch_idx, (inputs, targets) in enumerate(closeloader):
            progress.update(1)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        progress.close()
    acc = 100.0 * correct / total
    print("Test acc: %.3f%%" % acc)
    logger.info("Test acc: %.3f%%" % acc)

    early_stop += 1
    if acc > best_acc:
        print('==> Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, os.path.join(args.save_path, "best.pth"))
        print('==> Model saved')
        best_acc = acc
        early_stop = 0
    print("Early stop: ", early_stop)
    logger.info("Early stop: %d" % early_stop)

    if (epoch + 1) % 50 == 0:
        state = {"net": net.state_dict(), "epoch": epoch}
        torch.save(state, os.path.join(args.save_path, f"Epoch{epoch}.pth"))
    
    if early_stop == args.early_stop:
        print("==> Early stop")
        break