
import os
import time
import random
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import TensorDataset, DataLoader

from config import parse_args
from models import Wide_ResNet
from models.mymodel import *
from data.myLoader import MyDataset


def create_unique_folder_name(base_folder_path):
    count = 0
    new_folder_name = base_folder_path
    while os.path.exists(new_folder_name):
        count += 1
        new_folder_name = f"{base_folder_path}({count})"
    return new_folder_name


args = parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = args.device
print(vars(args))

# ======================================== Data prepare ======================================== #
print('\n ==> Preparing data')
mydataset = MyDataset(args.dataset, args.input_size, args.known_percent)

traindataX, traindataY = mydataset.traindataX, mydataset.traindataY
trainset = TensorDataset(traindataX, traindataY)
trainloader = DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

testclosedataX, testclosedataY = mydataset.testclosedataX, mydataset.testclosedataY
closeset = TensorDataset(testclosedataX, testclosedataY)
closerloader = DataLoader(
    closeset, batch_size=args.batch_size, shuffle=True, num_workers=4)

testopendataX, testopendataY = mydataset.testopendataX, mydataset.testopendataY
openset = TensorDataset(testopendataX, testopendataY)
openloader = DataLoader(openset, batch_size=args.batch_size, shuffle=True, num_workers=4)

rootDir = os.path.join(args.log_root, args.dataset)
save_path = f"{args.backbone}-Lr{args.lr}-Seed{args.seed}"
args.save_path = os.path.join(rootDir, save_path)
os.makedirs(args.log_root, exist_ok=True)
os.makedirs(rootDir, exist_ok=True)
os.makedirs(args.save_path, exist_ok=True)

# ======================================== Log ======================================== #
outlogPath = os.path.join(args.save_path,  "proser")
os.makedirs(outlogPath, exist_ok=True)
t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
log = outlogPath + "/" + t + "_proser" + '.txt'
logging.basicConfig(format='%(message)s', level=logging.INFO, filename=log, filemode='w')
logger = logging.getLogger(__name__)
argsDict = args.__dict__
for eachArg, value in argsDict.items():
    logger.info(eachArg + ':' + str(value))
logger.info("="*50)

# ======================================== Load model ======================================== #
if args.backbone == "WideResnet":
    net = Wide_ResNet(28, 10, 0.3, mydataset.classnum_known)
elif args.backbone == "Resnet50":
    net = models.resnet50(weights='DEFAULT')
    num_ftrs = net.fc.in_features
    net.fc = nn.Sequential(nn.Linear(num_ftrs, mydataset.classnum_known))
net = net.to(device)

print(' ==> Resuming from checkpoint..')
assert os.path.isdir(args.save_path), 'Error: no checkpoint directory found!'
modelname = "best.pth"
checkpoint = torch.load(os.path.join(args.save_path, modelname))
net.load_state_dict(checkpoint['net'])
net.clf2 = nn.Linear(2048, args.dummynumber)
net = net.to(device)

# ======================================== Train PROSER ======================================== #
FineTune_MAX_EPOCH = 500
best_finetuneacc = 0.0
early_stop = 0
for finetune_epoch in range(FineTune_MAX_EPOCH):
    # -------------------- Train -------------------- #
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr * 0.01, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), lr=args.lr * 0.0001, momentum=0.9, weight_decay=1e-4)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha = args.alpha

    progress = tqdm(total=len(trainloader), ncols=100, desc='PROER Train Epoch({})'.format(finetune_epoch))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        progress.update(1)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        totallenth = len(inputs)
        halflenth = int(len(inputs)/2)
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()

        prehalfinputs, prehalflabels = inputs[:halflenth], targets[:halflenth]
        laterhalfinputs, laterhalflabels = inputs[halflenth:], targets[halflenth:]

        index = torch.randperm(prehalfinputs.size(0)).to(device)
        pre2embeddings = pre2block(args.backbone, net, prehalfinputs)
        mixed_embeddings = beta * pre2embeddings + (1 - beta) * pre2embeddings[index]
        prehalfoutput = torch.cat((latter2blockclf1(args.backbone, net, mixed_embeddings),
                                   latter2blockclf2(args.backbone, net, mixed_embeddings)), 1)
        
        lateroutputs = net(laterhalfinputs)
        dummylogit = dummypredict(args.backbone, net, laterhalfinputs)
        latterhalfoutput = torch.cat((lateroutputs, dummylogit), 1)

        maxdummy, _ = torch.max(dummylogit.clone(), dim=1)
        maxdummy = maxdummy.view(-1, 1)
        dummpyoutputs = torch.cat((lateroutputs.clone(), maxdummy), dim=1)
        for i in range(len(dummpyoutputs)):
            nowlabel = laterhalflabels[i]
            dummpyoutputs[i][nowlabel] = -1e9
        dummytargets = torch.ones_like(laterhalflabels) * mydataset.classnum_known

        outputs = torch.cat((prehalfoutput, latterhalfoutput), 0)

        """
            - Loss1: deem mixup features to be unknown class
            - Loss2: maintain the ability to classify close-set classes
            - Loss3: forcing the dummy classifier to output the second-largest probability
        """
        loss1 = criterion(prehalfoutput, 
                          (torch.ones_like(prehalflabels) * mydataset.classnum_known).long().to(device))
        loss2 = criterion(latterhalfoutput, laterhalflabels)
        loss3 = criterion(dummpyoutputs, dummytargets)
        loss = 0.01 * loss1 + args.lamda1 * loss2 + args.lamda2 * loss3

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, known_predicted = latterhalfoutput.max(1)
        total += laterhalflabels.size(0)
        correct += known_predicted.eq(laterhalflabels).sum().item()
    progress.close()
    print("Train Loss: %.3f; Train acc: %.3f%%; L1: %.3f, L2: %.3f" % 
          (train_loss/(batch_idx + 1), 100.0 * correct / total, loss1.item(), loss2.item()))
    logger.info("Train Loss: %.3f; Train acc: %.3f%%; L1: %.3f, L2: %.3f" % 
          (train_loss/(batch_idx + 1), 100.0 * correct / total, loss1.item(), loss2.item()))
    
    # -------------------- Val -------------------- #
    if (finetune_epoch + 1) % 5 == 0:
        net.eval()
        CONF_AUC = True
        auclist1 = []
        linspace = [0]
        # init 
        closelogits = torch.zeros((len(closeset), mydataset.classnum_known + 1)).to(device)
        openlogits = torch.zeros((len(openset), mydataset.classnum_known + 1)).to(device)
        with torch.no_grad():
            progress = tqdm(total=len(openloader), ncols=100, desc='Test closeset Epoch({})'.format(finetune_epoch))
            for batch_idx, (inputs, targets) in enumerate(openloader):
                progress.update(1)
                inputs, targets = inputs.to(device), targets.to(device)
                batchnum = len(targets)
                logits = net(inputs)
                dummylogit = dummypredict(args.backbone, net, inputs)
                maxdummylogit, _ = torch.max(dummylogit, 1)
                maxdummylogit = maxdummylogit.view(-1, 1)
                totallogits = torch.cat((logits, maxdummylogit), dim=1)
                closelogits[batch_idx * batchnum:batch_idx * batchnum + batchnum,:] = totallogits 
            progress.close()
        
            progress = tqdm(total=len(openloader), ncols=100, desc='Test openset Epoch({})'.format(finetune_epoch))
            for batch_idx, (inputs, targets) in enumerate(openloader):
                progress.update(1)
                inputs, targets = inputs.to(device), targets.to(device)
                batchnum = len(targets)
                logits = net(inputs)
                dummylogit = dummypredict(args.backbone, net, inputs)
                maxdummylogit, _ = torch.max(dummylogit, 1)
                maxdummylogit = maxdummylogit.view(-1, 1)
                totallogits = torch.cat((logits, maxdummylogit), dim=1)
                openlogits[batch_idx * batchnum:batch_idx * batchnum + batchnum,:] = totallogits 
            progress.close()
        Logitsbatchsize = 200

        for biasitem in linspace:
            if CONF_AUC:
                for temperature in [1024.0]:
                    closeconf = np.array([])
                    openconf = np.array([])
                    closeiter=int(len(closelogits) / Logitsbatchsize)
                    openiter = int(len(openlogits) / Logitsbatchsize)

                    for batch_idx  in range(closeiter):
                        logitbatch = closelogits[batch_idx * Logitsbatchsize:batch_idx * Logitsbatchsize + Logitsbatchsize,:]
                        logitbatch[:,-1] = logitbatch[:,-1] + biasitem
                        embeddings = nn.functional.softmax(logitbatch / temperature, dim=1)
                        dummyconf = embeddings[:,-1].view(-1,1)
                        maxknownconf, _ = torch.max(embeddings[:, :-1], dim=1)
                        maxknownconf = maxknownconf.view(-1, 1)
                        conf = dummyconf - maxknownconf
                        closeconf = np.append(closeconf, conf.cpu().numpy())
                    closeconf = np.reshape(np.array(closeconf),(-1))
                    closelabel = np.ones_like(closeconf)

                    for batch_idx in range(openiter):
                        logitbatch = openlogits[batch_idx * Logitsbatchsize:batch_idx * Logitsbatchsize + Logitsbatchsize,:]
                        logitbatch[:, -1] = logitbatch[:, -1] + biasitem
                        embeddings = nn.functional.softmax(logitbatch / temperature, dim=1)
                        dummyconf = embeddings[:, -1].view(-1, 1)
                        maxknownconf, _ = torch.max(embeddings[:, :-1], dim=1)
                        maxknownconf = maxknownconf.view(-1, 1)
                        conf = dummyconf - maxknownconf
                        openconf = np.append(openconf, conf.cpu().numpy())
                    openconf = np.reshape(openconf, (-1))
                    openlabel = np.zeros_like(openconf)

                    totalbinary = np.hstack([closelabel, openlabel])
                    totalconf = np.hstack([closeconf, openconf])
                    auc1 = roc_auc_score(1 - totalbinary, totalconf)
                    auc2 = roc_auc_score(totalbinary, totalconf)
                    print('Temperature:', temperature, 'bias', biasitem, 'AUC_by_Delta_confidence', auc1)
                    logger.info("Temperature: %d; bias: %.3f; AUC_by_Delta_confidence: %.3f" % 
                                (temperature, biasitem, auc1))

                    fpr, tpr, thresholds = roc_curve(1 - totalbinary, totalconf)
                    auc_roc_dict = {'fpr': fpr, 'tpr': tpr}
                    np.set_printoptions(threshold = np.inf)
                    plt.title('roc-{}'.format(finetune_epoch))
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.plot(fpr, tpr, '--*b')
                    plt.savefig(os.path.join(args.save_path, 'Epoch{}_auc-roc.png'.format(finetune_epoch)))
                    plt.close()
                    auclist1.append(np.max([auc1, auc2]))

        finetuneacc = np.max(np.array(auclist1))
        print("Finetune acc:", finetuneacc)
        logger.info("Finetune acc: %.3f" % finetuneacc)

        early_stop += 1
        if finetuneacc > best_finetuneacc:
            best_finetuneacc = finetuneacc
            # state = {'net': net.state_dict(), 'epoch': finetune_epoch}
            # torch.save(state, os.path.join(args.save_path, 'Proser_Epoch' + str(finetune_epoch) + '.pth'))
            early_stop = 0
        
        print("Early stop: %d\n" % early_stop)
        logger.info("Early stop: %d\n" % early_stop)
        logger.info("="*50)
        if early_stop == 20:
            print("Early stop.")
            break