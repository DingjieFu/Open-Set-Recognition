import os
import argparse


def parse_args():
    projectPath = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description='OSR')
    parser.add_argument('--seed', default=2024,type=int,help='seed for reproduction')
    parser.add_argument('--log_root', default=projectPath+"/out", type=str, help='log saving path')
    parser.add_argument('--backbone', default='Resnet50', type=str, help='Backbone type.')
    parser.add_argument('--dataset', default='CUB', type=str, help='dataset configuration')    
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--nepoch', default=500, type=int, help='training epoches')
    parser.add_argument('--early_stop', default=10, type=int, help='early stop config')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size during training')
    parser.add_argument('--input_size', default=224, type=int, help='input image size')
    parser.add_argument('--dummynumber', default=1, type=int, help='number of dummy label.')
    parser.add_argument('--known_percent', default=0.6, type=float, help='N_known/N_all')
    parser.add_argument('--lamda1', default=1, type=float, help='trade-off between loss')
    parser.add_argument('--lamda2', default=1, type=float, help='trade-off between loss')
    parser.add_argument('--alpha', default=1, type=float, help='alpha value for beta distribution')
    parser.add_argument('--device', default='cuda:1', help='cpu/cuda:x')    
    args = parser.parse_args()
    return args
