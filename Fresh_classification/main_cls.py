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
from module.loss import CenterLoss
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
save_data = False
center = True
alpha = 1.1
lr_cent = 0.5

# Training option
learning_rate = 1e-5
momentum_ = 0.90
total_epoch = 500
model_name = 0 # 0:AlexNet1D, 1:ResNet, 2:VGG
optimizer_type='radam' #'sgd', 'adam', 'radam'
num_class = 3
type_name = 'cls'

# LR_schedule option
lr_schedule = False
step = 100

# Early Stopping
early_stop = False

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
data_all = [(data_x[3], criterion(int(data_x[1][1:]), int(data_x[0][3:])), data_x[4], data_x[5]) for data_x in data_meat] # (spectrum, [met_grade, ph_grade], pH, met)

# Train and Test Split
np.random.seed(42)
torch.manual_seed(10)

total_ix = list(range(len(data_all)))
train_rate = 0.7
train_ix = sorted(np.random.choice(total_ix, int(len(data_all)*train_rate), replace=False))
test_ix = sorted(np.array(list(set(total_ix) - set(train_ix))))

print(train_ix)

# Save the dataset
if save_data == True:
    for ix, ind in enumerate(train_ix):
        x = np.asarray(data_all[ind][0])
        y = np.array([data_all[ind][1][1]])
        tr_data = np.concatenate([x, y], axis=0).reshape(1,-1)
        if ix == 0:
            tr_all = tr_data
        else:
            tr_all = np.concatenate([tr_all, tr_data], axis=0)

    for ix, ind in enumerate(test_ix):
        x = np.asarray(data_all[ind][0])
        y = np.array([data_all[ind][1][1]])
        te_data = np.concatenate([x, y], axis=0).reshape(1,-1)
        if ix == 0:
            te_all = te_data
        else:
            te_all = np.concatenate([te_all, te_data], axis=0)

    dataset = {'train':tr_all, 'test':te_all}
    with open('./dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)

# Delete the label '1'
if num_class == 2:
    train_list = []
    test_list = []
    for ix in train_ix:
        if data_all[ix][1][1] != 1:
            train_list.append(ix)

            if data_all[ix][1][1] == 2:
                data_all[ix][1][1] =1

    for ix in test_ix:
        if data_all[ix][1][1] != 1:
            test_list.append(ix)

            if data_all[ix][1][1] == 2:
                data_all[ix][1][1] = 1

    train_ix, test_ix = train_list, test_list

# Dataset & DataLoder
class Meat_data(Dataset):
    def __init__(self, data_, index_list):
        self.data = data_
        self.index = index_list
        self.type = type

    def __getitem__(self, index):
        dat_x = np.expand_dims(np.asarray(self.data[self.index[index]][0]), axis=0)
        dat_x = torch.from_numpy(dat_x).float()

        dat_y = np.array(self.data[self.index[index]][1][1])
        dat_y = torch.from_numpy(dat_y).long()
        return dat_x, torch.tensor(-1000), dat_y

    def __len__(self):
        return len(self.index)

train_dataset = Meat_data(data_all, train_ix)
test_dataset = Meat_data(data_all, test_ix)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size = 512)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size = len(test_dataset))

## Model, Loss and Optimizer
# Select model
if model_name == 0:
    model = AlexNet1D(num_classes=num_class, result_emb=True)

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

if center == True:
    center_loss = CenterLoss(num_classes=num_class, feat_dim=num_class, use_gpu=True)
    params = list(model.parameters()) + list(center_loss.parameters())

else:
    params = model.parameters()

# Optimizer
if optimizer_type.lower() == 'sgd':
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum_, nesterov=True)
elif optimizer_type.lower() == 'adam':
    optimizer = optim.Adam(params, lr=learning_rate)
elif optimizer_type.lower() == 'radam':
    from utils.optimizer import RAdam
    optimizer = RAdam(params, lr=learning_rate)

# LR scheduler
if lr_schedule == True:
    from torch.optim import lr_scheduler
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.1)

## Train
if center == True:
    criterion = [cri_reg, cri_cls, center_loss]
else:
    criterion = [cri_reg, cri_cls]

option = {'gpu':gpu, 'type_name':type_name, 'print_epoch':20, 'num_class':num_class, 'reg':False, 'lambda1':0, 'lambda2':0,
          'alpha':alpha, 'lr_cent':lr_cent, 'lr':learning_rate}

for epoch in tqdm(range(total_epoch)):
    if lr_schedule == True:
        model = train(model, optimizer, epoch, train_loader, option, criterion, scheduler=scheduler)
    else:
        model = train(model, optimizer, epoch, train_loader, option, criterion, scheduler=None)

    if (epoch+1) % option['print_epoch'] == 0:
        model, te_loss, te_acc = test(model, epoch, test_loader, option, criterion)

## Save the model
os.makedirs('./result', exist_ok=True)
if multi == True:
    if center == True:
        torch.save({'model':model.module.state_dict(), 'loss':center_loss.state_dict()}, './result/model%d_epoch%d_center%d_%.2f.pt' %(model_name, total_epoch, center, te_acc))
    else:
        torch.save({'model':model.module.state_dict()}, './result/model%d_epoch%d_center%d_%.2f.pt' %(model_name, total_epoch, center, te_acc))
else:
    if center == True:
        torch.save({'model':model.state_dict(), 'loss':center_loss.state_dict()}, './result/model%d_epoch%d_center%d_%.2f.pt' %(model_name, total_epoch, center, te_acc))
    else:
        torch.save({'model':model.state_dict()}, './result/model%d_epoch%d_center%d_%.2f.pt' %(model_name, total_epoch, center, te_acc))

