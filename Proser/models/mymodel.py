import torch.nn.functional as F



def pre2block(backbone, net, x):
    if backbone == "WideResnet":
        out = net.conv1(x)
        out = net.layer1(out)
        out = net.layer2(out)
    elif backbone == 'Resnet50':
        out = net.conv1(x)
        out = net.bn1(out)
        out = net.relu(out)
        out = net.maxpool(out)
        out = net.layer1(out)
        out = net.layer2(out)
        return out


def dummypredict(backbone, net, x):
    if backbone == "WideResnet":
        out = net.conv1(x)
        out = net.layer1(out)
        out = net.layer2(out)
        out = net.layer3(out)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.clf2(out)
    elif backbone == 'Resnet50':
        out = net.conv1(x)
        out = net.bn1(out)
        out = net.relu(out)
        out = net.maxpool(out)
        out = net.layer1(out)
        out = net.layer2(out)
        out = net.layer3(out)
        out = net.layer4(out)
        out = net.avgpool(out)
        out = out.view(out.size(0), -1)
        out = net.clf2(out)
        return out


def latter2blockclf1(backbone, net, x):
    if backbone == "WideResnet":
        out = net.layer3(x)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.linear(out)
    elif backbone == 'Resnet50':
        out = net.layer3(x)
        out = net.layer4(out)
        out = net.avgpool(out)
        out = out.view(out.size(0), -1)
        out = net.fc(out)
        return out


def latter2blockclf2(backbone, net, x):
    if backbone == "WideResnet":
        out = net.layer3(x)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.clf2(out)
    elif backbone == 'Resnet50':
        out = net.layer3(x)
        out = net.layer4(out)
        out = net.avgpool(out)
        out = out.view(out.size(0), -1)
        out = net.clf2(out)
        return out