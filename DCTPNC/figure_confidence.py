import os
import pandas as pd
import numpy as np
import random

import torch

from hyperparameter import type, noise_ratio, ratio, forget_rate, worker_num

def calculate_cofidence(data=[]):
    # 计算所有标签的置信度
    sum_all = np.sum(np.exp(data))
    confidence = np.exp(data) / sum_all

    # 对置信度进行排序，获取最高和第二高的置信度
    sorted_confidence = np.sort(confidence)
    max_confidence = sorted_confidence[-1]
    second_max_confidence = sorted_confidence[-2]

    # 计算最高和第二高置信度的差
    confidence_diff = max_confidence - second_max_confidence

    # 返回最高置信度和置信度差
    return max_confidence,confidence_diff
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
clean_file = 'clean_' + type + '_' + str(noise_ratio) + '.csv'
noise_file = 'noise_' + type + '_' + str(noise_ratio) + '.csv'
dataset = pd.read_csv('../noise_label_data_' + type + '_（0.6,0.1）.csv')
attribute_num = len(dataset.iloc[0, :])
label = dataset.iloc[:, attribute_num - 1]
data = dataset.iloc[:, :attribute_num - 2 - worker_num]
column = []
for i in range(len(data.iloc[0, :])):
    column.append('feature' + str(i))
column.append("label")
column.append("truth_label")
feature_num = len(data.iloc[0, :])
clean_column = column.copy()
noise_column = column.copy()
worker_answer = dataset.iloc[:, attribute_num - 2 - worker_num:attribute_num - 2]
worker_answer_aggre = np.array(dataset.iloc[:, attribute_num - 2])
worker_answer_np = np.array(worker_answer).reshape([-1, worker_num])
def get_output():
    dataset1 = pd.read_csv('../noise_label_data_sys_（0.6,0.1）.csv')
    dataset2 = pd.read_csv('../noise_label_data_pair_（0.6,0.1）.csv')
    merged = pd.concat([dataset1["truth_label"], dataset2["truth_label"]])
    return len(merged.unique())
label_list = get_output()
confidence_matrix = np.zeros([len(worker_answer_aggre), label_list])
for i in range(len(worker_answer_aggre)):
    for j in range(worker_num):
        confidence_matrix[i][worker_answer_np[i][j]] = confidence_matrix[i][worker_answer_np[i][j]] + 1
confidence_matrix = confidence_matrix.astype(np.float32) / worker_num

confidence=np.zeros([len(confidence_matrix)])
for i in range(len(confidence_matrix)):
    X , Y  = calculate_cofidence(confidence_matrix[i])
    confidence[i]=X+Y
confidence_ind_sort=np.argsort(confidence)

clean_ind=confidence_ind_sort[int(len(confidence)*(1-ratio)):]
noise_ind=confidence_ind_sort[:int(len(confidence)*(1-ratio))]
data = np.array(data)
label = np.array(label)
clean_data = data[clean_ind]
noise_data = data[noise_ind]
clean_label = worker_answer_aggre[clean_ind].reshape([-1, 1])
noise_label = worker_answer_aggre[noise_ind].reshape([-1, 1])
clean_label_r = label[clean_ind].reshape([-1, 1])
noise_label_r = label[noise_ind].reshape([-1, 1])

clean_data = np.hstack((clean_data, clean_label, clean_label_r))
noise_data = np.hstack((noise_data, noise_label, noise_label_r))

clean_ratio = np.sum(clean_data[:, feature_num] == clean_data[:, feature_num + 1]) / len(clean_data)
noise_ratio = np.sum(noise_data[:, feature_num] == noise_data[:, feature_num + 1]) / len(noise_data)
error_ratio = (np.sum(clean_data[:, feature_num] == clean_data[:, feature_num + 1]) + np.sum(noise_data[:, feature_num] == noise_data[:, feature_num + 1])) / (len(clean_data) + len(noise_data))

clean_column[0] = str(clean_ratio)
clean_column[1] = str(error_ratio)
noise_column[0] = str(noise_ratio)
noise_column[1] = str(error_ratio)
clean_data = pd.DataFrame(clean_data, columns=clean_column)
noise_data = pd.DataFrame(noise_data, columns=noise_column)
clean_data.to_csv(clean_file, index=False)
noise_data.to_csv(noise_file, index=False)
print(f'DCTPNC噪声筛选准确率：{1 - noise_ratio}')