import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from module.model import AlexNet1D
from module.trainer import train, test
from tqdm import tqdm
import os
import argparse

## ArgParse
arg = argparse.ArgumentParser()
arg.add_argument('--gpu', default='1', type=str)
arg.add_argument('--type', default='reg', type=str)
args = arg.parse_args()

## GPU option
if torch.cuda.is_available():
    gpu_num = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
    if len(gpu_num.split(',')) > 1:
        multi = True
    else:
        multi = False
    gpu = True
else:
    gpu = False

## Option
# Training option
class_size = 3
learning_rate = 1e-5
momentum_ = 0.90
total_epoch = 300
model_name = 0 # 0:AlexNet1D, 1:ResNet, 2:VGG
optimizer_type='RAdam' #'sgd', 'adam', 'radam'
num_class = 3

type_name = 'both' #'reg', 'cls', 'both'

# LR_schedule option
lr_schedule = False
step = 1

# Regularization coefficient
reg = False
lambda1 = 0.
lambda2 = 0.

# Early Stopping
early_stop = False

# Knowledge Distillation coefficient
teacher = False
t = 3
lambda_k = 0.3

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
data_tot = [(data_x[3], criterion(int(data_x[1][1:]), int(data_x[0][3:])), data_x[4], data_x[5]) for data_x in data_meat] # (spectrum, [met_grade, ph_grade], pH, met)

# Train and Test Split
total_ix = list(range(len(data_tot)))
train_rate = 0.7
train_ix = sorted(np.random.choice(total_ix, int(len(data_tot)*train_rate), replace=False))
test_ix = sorted(np.array(list(set(total_ix) - set(train_ix))))

# Dataset & DataLoder
class Meat_data(Dataset):
    def __init__(self, data_, index_list, type):
        self.data = data_
        self.index = index_list
        self.type = type
    def __getitem__(self, index):
        dat_x = np.expand_dims(np.asarray(self.data[self.index[index]][0]), axis=0)
        dat_x = torch.from_numpy(dat_x).float()

        if self.type == 'reg':
            dat_y = np.array(self.data[self.index[index]][2])
            dat_y = torch.from_numpy(dat_y).float()
            return dat_x, dat_y, torch.tensor(-1000)

        elif self.type == 'cls':
            dat_y = np.array(self.data[self.index[index]][1][1])
            dat_y = torch.from_numpy(dat_y).long()
            return dat_x, torch.tensor(-1000), dat_y

        elif self.type == 'both':
            reg_y = torch.from_numpy(np.array(self.data[self.index[index]][2])).float()
            cls_y = torch.from_numpy(np.array(self.data[self.index[index]][1][1])).long()
            return dat_x, reg_y, cls_y

    def __len__(self):
        return len(self.index)

train_dataset = Meat_data(data_tot, train_ix, type=type_name)
test_dataset = Meat_data(data_tot, test_ix, type=type_name)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size = 512)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size = len(test_dataset))

## Model, Loss and Optimizer
# Select model
if model_name == 0:
    model = AlexNet1D(num_classes=class_size)

# Type
model.up_type(type_name)

# Select GPU
if gpu == True:
    if multi == True:
        model = nn.DataParallel(model)
    model = model.cuda()

# Loss_type
if type_name == 'reg':
    cri_reg = nn.MSELoss()
    cri_cls = None

elif type_name == 'cls':
    cri_cls = nn.CrossEntropyLoss()
    cri_reg = None

else:
    cri_reg = nn.MSELoss()
    cri_cls = nn.CrossEntropyLoss()

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
criterion = [cri_reg, cri_cls]
option = {'gpu':gpu, 'type_name':type_name, 'reg':reg, 'lambda1':lambda1, 'lambda2':lambda2, 'print_epoch':20, 'num_class':num_class}

for epoch in tqdm(range(total_epoch)):
    if option['reg'] == True:
        model = train(model, optimizer, epoch, train_loader, option, criterion, scheduler=scheduler)
    else:
        model = train(model, optimizer, epoch, train_loader, option, criterion, scheduler=None)

    if (epoch+1) % option['print_epoch'] == 0:
        model, te_loss, te_acc = test(model, epoch, test_loader, option, criterion)

        if early_stop == True:
            pass
