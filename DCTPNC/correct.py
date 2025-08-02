import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from DataSet import DataSet
from model import MLPNet
import numpy as np
from hyperparameter import type,noise_ratio,ratio,forget_rate,worker_num,work_num
def set_seeds(seed):
    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for Python's built-in random module
    random.seed(seed)

    # Set the random seed for PyTorch
    torch.manual_seed(seed)
    # If using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seeds(7)
#生成dataloder

dataset=pd.read_csv(f'noise_{type}_{noise_ratio}.csv')
attribute_num=len(dataset.iloc[0,:])


label=dataset.iloc[:,attribute_num-2]
data=dataset.iloc[:,:attribute_num-2]
truth_label=dataset.iloc[:,attribute_num-1]
input=data.shape[1]
merged = pd.concat([dataset["truth_label"],dataset["label"]])
output=merged.nunique()
data=torch.tensor(np.array(data),dtype=torch.float32)
label=torch.tensor(np.array(label),dtype=torch.float32)
truth_label=torch.tensor(np.array(truth_label),dtype=torch.float32)
train_data = torch.utils.data.TensorDataset(data, label,truth_label)
train_loader = DataLoader(train_data, batch_size=10, shuffle=False, drop_last=True)
#读取模型
fclean=torch.load('fclean_confidence_'+type+'_'+str(noise_ratio)+'.pt').eval().cuda()
fnoise1=torch.load('fnoise1_confidence_'+type+'_'+str(noise_ratio)+'.pt').eval().cuda()
fnoise2=torch.load('fnoise2_confidence_'+type+'_'+str(noise_ratio)+'.pt').eval().cuda()
#纠正标签


correct_predictions = 0
total_predictions = 0
clean=0
noise1=0
noise2=0
for data, label, truth_label in train_loader:
    data= Variable(data).float().cuda()
    pre_clean = fclean(data)
    pre_noise1 = fnoise1(data)
    pre_noise2 = fnoise2(data)

    # 获取预测标签，即每行得分最高的类别
    pre_clean_label = torch.argmax(pre_clean, dim=1)
    pre_noise1_label = torch.argmax(pre_noise1, dim=1)
    pre_noise2_label = torch.argmax(pre_noise2, dim=1)

    for i in range(label.shape[0]):
        total_predictions += 1
        if pre_clean_label[i] == truth_label[i]:
            clean += 1
        if pre_noise1_label[i] == truth_label[i]:
            noise1 += 1
        if pre_noise2_label[i] == truth_label[i]:
            noise2 += 1
        # 预测结果一致的情况下，将预测标签与真实标签进行比较
        if pre_clean_label[i] == pre_noise1_label[i] == pre_noise2_label[i]:
            if pre_clean_label[i] == truth_label[i].cuda():
                correct_predictions += 1
        else:
            # 如果预测结果不一致，那么比较原始标签和真实标签
            if label[i] == truth_label[i]:
                correct_predictions += 1




dataset=pd.read_csv(f'clean_{type}_{noise_ratio}.csv')
attribute_num=len(dataset.iloc[0,:])


label=dataset.iloc[:,attribute_num-2]
data=dataset.iloc[:,:attribute_num-2]
truth_label=dataset.iloc[:,attribute_num-1]
input=data.shape[1]
merged = pd.concat([dataset["truth_label"],dataset["label"]])
output=merged.nunique()
data=torch.tensor(np.array(data),dtype=torch.float32)
label=torch.tensor(np.array(label),dtype=torch.float32)
truth_label=torch.tensor(np.array(truth_label),dtype=torch.float32)
train_data = torch.utils.data.TensorDataset(data, label,truth_label)
train_loader = DataLoader(train_data, batch_size=10, shuffle=False, drop_last=True)
#读取模型
fclean=torch.load('fclean_confidence_'+type+'_'+str(noise_ratio)+'.pt').eval().cuda()
fnoise1=torch.load('fnoise1_confidence_'+type+'_'+str(noise_ratio)+'.pt').eval().cuda()
fnoise2=torch.load('fnoise2_confidence_'+type+'_'+str(noise_ratio)+'.pt').eval().cuda()
#纠正标签


correct_predictions2 = 0
total_predictions2 = 0
clean=0
noise1=0
noise2=0
for data, label, truth_label in train_loader:
    data= Variable(data).float().cuda()
    pre_clean = fclean(data)
    pre_noise1 = fnoise1(data)
    pre_noise2 = fnoise2(data)

    # 获取预测标签，即每行得分最高的类别
    pre_clean_label = torch.argmax(pre_clean, dim=1)
    pre_noise1_label = torch.argmax(pre_noise1, dim=1)
    pre_noise2_label = torch.argmax(pre_noise2, dim=1)

    for i in range(label.shape[0]):
        total_predictions2 += 1
        if pre_clean_label[i] == truth_label[i]:
            clean += 1
        if pre_noise1_label[i] == truth_label[i]:
            noise1 += 1
        if pre_noise2_label[i] == truth_label[i]:
            noise2 += 1
        # 预测结果一致的情况下，将预测标签与真实标签进行比较
        if pre_clean_label[i] == pre_noise1_label[i] == pre_noise2_label[i]:
            if pre_clean_label[i] == truth_label[i].cuda():
                correct_predictions2 += 1
        else:
            # 如果预测结果不一致，那么比较原始标签和真实标签
            if label[i] == truth_label[i]:
                correct_predictions2 += 1



acc=(correct_predictions+correct_predictions2)/(total_predictions2+total_predictions)
print("DCTPNC标签质量："+f"{acc:.4f}")