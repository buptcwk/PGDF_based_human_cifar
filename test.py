import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture

from torch.utils.data import DataLoader, Dataset

import dataloader_cifar as dataloader
import dataloader_easy 
from PreResNet import *
from preset_parser import *
import pickle
from data.datasets import input_dataset2

parser = argparse.ArgumentParser()
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')


args = parser.parse_args()

detect_file = args.noise_type + "_detection.npy"

noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]

num_classes = 10 if args.dataset =="cifar10" else 100

def create_model(devices=[0]):
    model = ResNet18(num_classes=num_classes)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=devices).cuda()
    return model


net1 = create_model()
net2 = create_model()
state_dict = torch.load(f"./{args.dataset}_{args.noise_type}best.pth.tar")

net1.load_state_dict(state_dict["net1"])
net2.load_state_dict(state_dict["net2"])

# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else: 
        raise NameError(f'Undefined dataset {args.dataset}')


train_dataset = input_dataset2(args.dataset,args.noise_type, args.noise_path, is_human = True, val_ratio = 0.0)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                  batch_size = 64,
                                  num_workers=args.num_workers,
                                  shuffle=False)



net1.eval()    # Change model to 'eval' mode.
net2.eval() 

detection = np.zeros(50000)

for images, labels, index in train_loader:
    images = Variable(images).cuda()
    outputs1 = net1(images)
    outputs2 = net2(images)
    outputs = outputs1 + outputs2
    _, pred = torch.max(outputs, 1)

    for i in range(index.shape[0]):
        if pred[i] == labels[i]:
            detection[index[i]] = False
        else:
            detection[index[i]] = True


np.save(detect_file,detection)

