import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from DataSet import DataSet
from model import MLPNet
import argparse, sys
import numpy as np
import datetime
import shutil
from hyperparameter import type,noise_ratio,ratio,forget_rate,worker_num,work_num
def evaluate(test_loader, model1, model2):

    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels, right_label in test_loader:

        data = Variable(data).float().cuda()
        logits1 = model1(data)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == right_label).sum()
    acc1 = 100 * float(correct1) / float(total1)
    return acc1
def get_output():
    dataset1 = pd.read_csv('../noise_label_data_sys_（0.6,0.1）.csv')
    dataset2 = pd.read_csv('../noise_label_data_pair_（0.6,0.1）.csv')
    merged = pd.concat([dataset1["truth_label"], dataset2["truth_label"]])
    return len(merged.unique())
output=get_output()
dataset=pd.read_csv('../noise_label_data_'+type+'_（0.6,0.1）.csv')
merged = pd.concat([dataset["truth_label"],dataset["noise_label_mv"]])

dataset=pd.read_csv('clean_'+type+'_'+str(noise_ratio)+'.csv')
attribute_num=len(dataset.iloc[0,:])
label=dataset.iloc[:,attribute_num-2]
data=dataset.iloc[:,:attribute_num-2]
truth_label=dataset.iloc[:,attribute_num-1]
input=data.shape[1]

data=torch.tensor(np.array(data),dtype=torch.float32)
label=torch.tensor(np.array(label),dtype=torch.float32)
truth_label=torch.tensor(np.array(truth_label),dtype=torch.float32)
train_data = torch.utils.data.TensorDataset(data, label,truth_label)
train_loader = DataLoader(train_data, batch_size=20, shuffle=True, drop_last=True)
model_path='fclean_confidence_'+type+'_'+str(noise_ratio)+'.pt'
clf1 = MLPNet(input=input, output=output)
clf1.cuda()
optimizer1 = torch.optim.Adam(clf1.parameters(), lr=0.0001)
for epoch in range(1, 150):
    clf1.train()
    for i, (data, labels,truth_label) in enumerate(train_loader):

        labels = Variable(labels).long().cuda()
        data = Variable(data).float().cuda()
        # Forward + Backward + Optimize
        logits1 = clf1(data)
        loss=F.cross_entropy(logits1,labels)
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()
    acc = evaluate(train_loader, clf1, clf1)

torch.save(clf1,model_path)