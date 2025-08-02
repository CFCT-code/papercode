import random

import pandas as pd
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from hyperparameter import type,noise_ratio,ratio,forget_rate,worker_num,work_num
from DataSet import DataSet
from model import MLPNet
import numpy as np
from loss_coteachingplus2 import loss_coteaching, loss_coteaching_plus


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
def get_output():
    dataset1 = pd.read_csv('../noise_label_data_sys_（0.6,0.1）.csv')
    dataset2 = pd.read_csv('../noise_label_data_pair_（0.6,0.1）.csv')
    merged = pd.concat([dataset1["truth_label"], dataset2["truth_label"]])
    return len(merged.unique())
output=get_output()


learning_rate=0.0001
n_epoch=200
epoch_decay_start=20
init_epoch=50
num_gradual=10
exponent=1

model_path='fclean_confidence_'+type+'_'+str(noise_ratio)+'.pt'
fclean=torch.load(model_path)
#生成dataloder
dataset=pd.read_csv('../noise_label_data_'+type+'_（0.6,0.1）.csv')
attribute_num=len(dataset.iloc[0,:])
label=dataset.iloc[:,attribute_num-2]
data=dataset.iloc[:,:attribute_num-2-work_num]
truth_label=dataset.iloc[:,attribute_num-1]
ind=np.arange(0,len(data))
input=data.shape[1]

data=torch.tensor(np.array(data),dtype=torch.float32)
label=torch.tensor(np.array(label),dtype=torch.float32)
ind=torch.tensor(np.array(ind),dtype=torch.int)
truth_label=torch.tensor(np.array(truth_label),dtype=torch.float32)
train_data = torch.utils.data.TensorDataset(data, label,truth_label,ind)
train_loader = DataLoader(train_data, batch_size=20, shuffle=True, drop_last=True)
# Adjust learning rate and betas for Adam Optimizer
mom1 = 0.9
mom2 = 0.1
alpha_plan = [learning_rate] * n_epoch
beta1_plan = [mom1] * n_epoch
for i in range(epoch_decay_start, n_epoch):
    alpha_plan[i] = float(n_epoch - i) / (n_epoch - epoch_decay_start) * learning_rate
    beta1_plan[i] = mom2
def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        param_group['betas'] = (beta1_plan[epoch], 0.999)
rate_schedule = np.ones(n_epoch)*forget_rate
rate_schedule[:num_gradual] = np.linspace(0, forget_rate**exponent, num_gradual)
def train_plus(disaggre_ind,model,optimizer):
    disaggre_ind = [int(i.item()) for i in disaggre_ind]
    disaggre_ind=disaggre_ind[0]
    label = dataset.iloc[disaggre_ind:, attribute_num - 2]
    data = dataset.iloc[disaggre_ind:, :attribute_num - 2 - work_num]
    truth_label = dataset.iloc[disaggre_ind:, attribute_num - 1]
    data = torch.tensor(np.array(data), dtype=torch.float32)
    label = torch.tensor(np.array(label), dtype=torch.float32)
    truth_label = torch.tensor(np.array(truth_label), dtype=torch.float32)
    train_data = torch.utils.data.TensorDataset(data, label, truth_label)
    train_loader_dis = DataLoader(train_data, batch_size=10, shuffle=True, drop_last=True)
    for i, (data, labels, right_label) in enumerate(train_loader_dis):
        labels = Variable(labels).long().cuda()
        data = Variable(data).float().cuda()
        logits = model(data)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
# Train the Model
def train(train_loader, epoch, model1, optimizer1, model2, optimizer2):
    train_total = 0
    train_correct = 0
    train_total2 = 0
    train_correct2 = 0
    disaggre_ind1=[]
    disaggre_ind2 = []
    err_find=0
    for i, (data, labels,right_label,ind) in enumerate(train_loader):

        labels = Variable(labels).long().cuda()
        data = Variable(data).float().cuda()
        # Forward + Backward + Optimize
        logits1 = model1(data)
        train_total += labels.shape[0]
        logits2 = model2(data)
        train_total2 += labels.shape[0]
        logits3 = fclean(data)
        # loss_1, loss_2 ,num= loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], right_label)

        if epoch < init_epoch:
            loss_1, loss_2, num = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], right_label,logits3)
        else:
            loss_1, loss_2, num = loss_coteaching_plus(logits1, logits2, labels, rate_schedule[epoch], right_label,logits3)
        err_find += num
        # loss_1=F.cross_entropy(logits1,labels)
        # loss_2=F.cross_entropy(logits2,labels)
        optimizer1.zero_grad()
        loss_1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss_2.backward()
        optimizer2.step()
    train_acc1 = float(err_find) / (float(train_total) * (rate_schedule[epoch]))

    return train_acc1
# Evaluate the Model
def evaluate(test_loader, model1, model2):

    model1.eval()  # Change model to 'eval' mode.
    correct1 = 0
    total1 = 0
    for data, labels,right_label,ind in test_loader:

        data = Variable(data).float().cuda()
        logits1 = model1(data)
        outputs1 = F.softmax(logits1, dim=1)
        _, pred1 = torch.max(outputs1.data, 1)
        total1 += labels.size(0)
        correct1 += (pred1.cpu() == right_label).sum()
    acc1 = 100 * float(correct1) / float(total1)
    return acc1
def main():

    clf1 = MLPNet(input=input,output=output)
    clf1.cuda()
    optimizer1 = torch.optim.Adam(clf1.parameters(), lr=learning_rate)
    clf2 = MLPNet(input=input, output=output)
    clf2.cuda()
    optimizer2 = torch.optim.Adam(clf2.parameters(), lr=learning_rate)

    for epoch in range(1, n_epoch):
        # train models
        clf1.train()
        clf2.train()

        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)

        err_find_acc= train(train_loader, epoch, clf1, optimizer1, clf2, optimizer2)

        acc=evaluate(train_loader,clf2,clf1)



    torch.save(clf1,'fnoise1_confidence_'+type+'_'+str(noise_ratio)+'.pt')
    torch.save(clf2,'fnoise2_confidence_'+type+'_'+str(noise_ratio)+'.pt')
if __name__ == '__main__':
    main()