import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import AlexNet1D
from tqdm import tqdm
import os

## GPU option
if torch.cuda.is_available():
    gpu_num = '1,2'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
    if len(gpu_num.split(',')) > 1:
        multi = True
    else:
        multi = False
    gpu = True
else:
    gpu = False

## Load the Data
'''
Data Path: './Data_pre/Meat_data.pkl'
Data Format : [(day1, s1, [ Wavelength ], [ spectrum ], pH, met), (day1, s2, [ ~~ ], [ ~~ ], pH, met), ..., (day33, s78, [ ~~ ], [ ~~ ], pH, met)
Data Type : text files, excel --> npz
Data Length : 2574 (78개 샘플 x 33 days)
'''
# Spectrum Data
data_meat = np.load('../Data_pre/Meat_data.npy', allow_pickle=True)
# Label
label_meat = dict(np.load('../Data_pre/Meat_label.npz'))

# Criterion for freshness
def criterion(sample_num, day):
    answer = label_meat[str(sample_num)]

    if day >= answer[1]:
        met_grade = 2
    elif day >= answer[0]:
        met_grade  = 1
    else:
        met_grade= 0

    if day >= answer[3]:
        ph_grade = 2
    elif day >= answer[2]:
        ph_grade = 1
    else:
        ph_grade = 0

    return [met_grade, ph_grade]

# Data of X and Y Variable
data_tot = [(data_x[3], criterion(int(data_x[1][1:]), int(data_x[0][3:]))) for data_x in data_meat]

# Train and Test Split
total_ix = list(range(len(data_tot)))
train_rate = 0.7
train_ix = sorted(np.random.choice(total_ix, int(len(data_tot)*train_rate), replace=False))
test_ix = sorted(np.array(list(set(total_ix) - set(train_ix))))

# Dataset & DataLoder
class Meat_data(Dataset):
    def __init__(self, data_, index_list):
        self.data = data_
        self.index = index_list

    def __getitem__(self, index):
        dat_x = np.expand_dims(np.asarray(self.data[self.index[index]][0]), axis=0)
        dat_y = np.array(self.data[self.index[index]][1][1])

        dat_x = torch.from_numpy(dat_x).float()
        dat_y = torch.from_numpy(dat_y).long()

        return dat_x, dat_y

    def __len__(self):
        return len(self.index)

train_dataset = Meat_data(data_tot, train_ix)
test_dataset = Meat_data(data_tot, test_ix)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size = 32)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size = len(test_dataset))

## Option
# Training option
class_size = 3
learning_rate = 1e-5
momentum_ = 0.90
total_epoch = 101
model = 0 # 0:AlexNet1D, 1:ResNet, 2:VGG
optimizer_type='RAdam'

# LR_schedule option
lr_schedule = False
step = 1

# Regularization coefficient
lambda1 = 0.
lambda2 = 0.

# Knowledge Distillation coefficient
teacher = False
t = 3
lambda_k = 0.3

## Model, Loss and Optimizer
if gpu == True:
    if multi == True:
        model = nn.DataParallel(AlexNet1D(num_classes=class_size))
    else:
        model = AlexNet1D(num_classes=class_size)

    model = model.cuda()

else:
    model = AlexNet1D(num_classes=class_size)

loss_cri = nn.CrossEntropyLoss()
# Optimizer
if optimizer_type.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momemtum=momentum_, nesterov=True)
elif optimizer_type.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif optimizer_type.lower() == 'radam':
    from utils.optimizer import RAdam
    optimizer = RAdam(model.parameters(), lr=learning_rate)

# LR scheduler
if lr_schedule == True:
    from torch.optim import lr_scheduler
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.1)

# Knowledge Distillation
if teacher == True:
    teacher_model = None

## Train
model.train()
for epoch in tqdm(range(total_epoch)):
    if lr_schedule == True:
        scheduler.step(epoch)

    loss_ = 0.
    acc_tr = 0.
    for ix, (tr_x, tr_y) in enumerate(train_loader):
        if gpu == True:
            tr_x = tr_x.cuda()
            tr_y = tr_y.cuda().view(-1)
        else:
            tr_x = tr_x.cpu()
            tr_y = tr_y.cpu().view(-1)

        l1_regularization, l2_regularization = torch.tensor(0.).cuda(), torch.tensor(0.).cuda()

        # Foward Propagation
        predicted = model(tr_x)
        for param in model.parameters():
            l1_regularization += lambda1 * torch.norm(param, 1)
            l2_regularization += lambda2 * torch.norm(param, 2)

        loss_tr = F.cross_entropy(predicted, tr_y) + l1_regularization + l2_regularization

        # Teacher & Student
        if teacher == True:
            predicted_teacher = teacher_model(tr_x)
            loss_knowledge = nn.KLDivLoss()(F.log_softmax(predicted / t, dim=1),
                                            F.softmax(predicted_teacher / t, dim=1))
            loss_total = (loss_tr.data * (1 - lambda_k) + loss_knowledge * lambda_k)
        else:
            loss_total = loss_tr

        loss_ += loss_tr.data

        _, predicted_l = torch.max(predicted.data, dim=1)
        predicted_l = predicted_l.view(-1)
        target_l = tr_y.data
        acc_tr += (torch.mean((predicted_l == target_l).float())) * 100

        # BackPropagation
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

    acc_tr = acc_tr / len(train_loader)
    loss_ = loss_ / len(train_loader)
    if epoch % 10 == 0:
        print('Epoch : %d, Train Acc : %2.2f, Train Loss : %.2f,' % (epoch, acc_tr.cpu(), loss_.cpu()))

    ## Test
    model.eval()
    with torch.no_grad():
        confusion = torch.zeros([class_size, class_size])

        acc_te = 0
        loss_te = 0
        for te_x, te_y in test_loader:
            if gpu == True:
                te_x = te_x.cuda()
                te_y = te_y.cuda().view(-1)
            else:
                te_x = te_x.cpu()
                te_y = te_y.cpu().view(-1)

            predicted = model(te_x)

            _, predicted_l = torch.max(predicted.data, 1)
            predicted_l = predicted_l.view(-1)

            acc_te = torch.mean((predicted_l == te_y).float()).cpu() * 100

            for (x, y) in zip(te_y, predicted_l):
                confusion[x, y] += 1

        if epoch % 10 == 0:
            print('Epoch : %d, Test Acc : %2.2f' % (epoch, acc_te))
            print(confusion)
