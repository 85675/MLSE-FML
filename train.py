from __future__ import print_function
import os

import numpy
import numpy as np
import pandas as pd
from PIL import Image
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from sklearn.model_selection import KFold,StratifiedKFold
import cv2
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader
import re
from torch.utils.data.sampler import WeightedRandomSampler
from utils import *
import os


class FML(nn.Module):
    def __init__(self, x_concat1, out_features, s=64):
        super(FML, self).__init__()
        self.s = s
        self.weight1 = Parameter(torch.FloatTensor(out_features, x_concat1))
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, x_concat1, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(x_concat1), F.normalize(self.weight1))
        datap = F.softmax(cosine, dim=1)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda:1')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        datapr = one_hot*datap
        predictedpr,_ = torch.max(datapr, 1)
        output1 = torch.zeros(cosine.size(), device='cuda:1')

        for i in range(cosine.shape[0]):
            phi = cosine[i] - (0.5/ (math.exp(predictedpr[i])))
            output1[i] = (one_hot[i] * phi) + ((1.0 - one_hot[i]) * cosine[i])
        output1 *= self.s

        return output1

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


# Model
net = load_model(model_name='resnet50', pretrain=True, require_grad=True)
netp = net
# # GPU
device = torch.device("cuda:1")
net.to(device)
classifier1 = FML(256, 2).to(device)
CELoss = nn.CrossEntropyLoss()
optimizer1 = optim.Adam([
    {'params': net.parameters(), 'lr': 2e-5},
],weight_decay=5e-4)

max_val_acc = 0
e1 = 0

train_loader = ""
val_loader = ""

for epoch in range(200):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    test_target= []
    test_data_predict = []
    test_data_predict_proba = []
    for inputs, label in tqdm(train_loader):
        inputs, targets = inputs.to(device), label.to(device)
        optimizer1.zero_grad()
        with autocast():
            output_concat = netp(inputs)
            output_concat = classifier1(output_concat, targets)
            concat_loss1 = CELoss(output_concat, targets)
        (concat_loss1).backward()
        optimizer1.step()

        #  training log
        data1 = F.softmax(output_concat, dim=1)
        _, predicted = torch.max(output_concat.data, 1)
        train_loss += concat_loss1.item()
        test_data_predict1 = predicted.cpu().numpy()
        test_data_predict = np.hstack([test_data_predict, test_data_predict1])
        test_data_predict_proba.extend(data1[:, 1].detach().cpu().numpy())
        test_target1 = targets.data.cpu().numpy()
        test_target = np.hstack([test_target, test_target1])

    test_data_confusion_matrix = confusion_matrix(test_target, test_data_predict)
    M = test_data_confusion_matrix
    print(test_data_confusion_matrix)

    test_data_accuracy_score = accuracy_score(test_target, test_data_predict)
    print(test_data_accuracy_score)

    sensitity = M[1, 1] / (M[1, 1] + M[1, 0])
    print(sensitity)

    specificity = M[0, 0] / (M[0, 1] + M[0, 0])
    print(specificity)

    FPR_test_data, TPR_test_data, threshold_test_data = roc_curve(test_target, test_data_predict_proba)
    test_data_roc_auc = auc(FPR_test_data, TPR_test_data)
    print(test_data_roc_auc)

    aucval = test(val_loader,net,classifier1)

    if aucval > max_val_acc:
        e1=epoch
        max_val_acc = aucval
        torch.save(net, '')
        torch.save(classifier1, '')
    if int(epoch) - int(e1) > 20:
        break