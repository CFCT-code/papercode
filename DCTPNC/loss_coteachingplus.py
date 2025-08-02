from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from numpy.testing import assert_array_almost_equal

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, truth_label,y_3):
    loss_3 = F.cross_entropy(y_3, t, reduction='none')

    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    loss_1=(loss_3+loss_1)/2
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    loss_2=(loss_3+loss_2)/2
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember].cpu()
    ind_2_update=ind_2_sorted[:num_remember].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])
    t = t.cpu()
    # 计算错误数据筛选正确的个数
    arr=torch.arange(0,len(t))
    result1=np.array([i for i in arr if i not in ind_1_update])
    result2 = np.array([i for i in arr if i not in ind_2_update])
    num_1 = torch.sum(t[result1] == truth_label[result1])
    num_2 = torch.sum(t[result2] == truth_label[result2])
    num = (num_2 + num_1) / 2
    err_num =len(t) - num_remember - num
    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember,err_num

def loss_coteaching_plus(logits, logits2, labels, forget_rate, truth_label,ind,y_3):
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()
    logical_disagree_id=np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            disagree_id.append(idx) 
            logical_disagree_id[idx] = True
    

    ind_disagree = np.asarray([i for i in logical_disagree_id if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0]==len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]


    if len(disagree_id) > 0:
        #计算num错误标签的数据
        y_1=logits
        y_2=logits2
        t=labels
        loss_3 = F.cross_entropy(y_3, t, reduction='none')
        loss_1 = F.cross_entropy(y_1, t, reduction='none')
        loss_1=(loss_3+loss_1)/2
        ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, t, reduction='none')
        loss_2=(loss_3+loss_2)/2
        ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember].cpu()
        ind_2_update = ind_2_sorted[:num_remember].cpu()
        if len(ind_1_update) == 0:
            ind_1_update = ind_1_sorted.cpu().numpy()
            ind_2_update = ind_2_sorted.cpu().numpy()
            num_remember = ind_1_update.shape[0]
        t = t.cpu()
        # 计算错误数据筛选正确的个数
        arr = torch.arange(0, len(t))
        result1 = np.array([i for i in arr if i not in ind_1_update])
        result2 = np.array([i for i in arr if i not in ind_2_update])

        num_1 = torch.sum(t[result1] == truth_label[result1])
        num_2 = torch.sum(t[result2] == truth_label[result2])
        num = (num_2 + num_1) / 2
        num = len(t) - num_remember - num
        ind_1_chosen=ind[ind_1_update]
        ind_2_chosen = ind[ind_2_update]
        return ind_1_chosen,ind_2_chosen
    else:
        return None, None


def loss_coteaching_plus2(logits, logits2, labels, forget_rate, truth_label):
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()
    logical_disagree_id = np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1):
        if p1 != pred2[idx]:
            disagree_id.append(idx)
            logical_disagree_id[idx] = True

    ind_disagree = np.asarray([i for i in logical_disagree_id if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0] == len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id]
        update_outputs2 = outputs2[disagree_id]

        # 计算num错误标签的数据
        y_1 = logits
        y_2 = logits2
        t = labels
        loss_1 = F.cross_entropy(y_1, t, reduction='none')

        ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = F.cross_entropy(y_2, t, reduction='none')

        ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update = ind_1_sorted[:num_remember].cpu()
        ind_2_update = ind_2_sorted[:num_remember].cpu()
        if len(ind_1_update) == 0:
            ind_1_update = ind_1_sorted.cpu().numpy()
            ind_2_update = ind_2_sorted.cpu().numpy()
            num_remember = ind_1_update.shape[0]
        t = t.cpu()
        # 计算错误数据筛选正确的个数
        arr = torch.arange(0, len(t))
        result1 = np.array([i for i in arr if i not in ind_1_update])
        result2 = np.array([i for i in arr if i not in ind_2_update])

        num_1 = torch.sum(t[result1] == truth_label[result1])
        num_2 = torch.sum(t[result2] == truth_label[result2])
        num = (num_2 + num_1) / 2
        num = len(t) - num_remember - num
        loss_1, loss_2, _ = loss_coteaching(update_outputs, update_outputs2, update_labels, forget_rate,
                                            truth_label[disagree_id])
    else:
        loss_1, loss_2, num = loss_coteaching(logits, logits2, labels, forget_rate, truth_label)
    return loss_1, loss_2, num

