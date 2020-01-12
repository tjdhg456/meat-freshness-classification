import numpy as np
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from module.model import AlexNet1D
from module.trainer import ad_train, test
from module.loss import AD_CenterLoss
import torch.nn as nn
import torch.optim as optim
import utils
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader

# Option
np.random.seed(42)
torch.manual_seed(10)

model_name = 0
pt_name = './result/model0_epoch500_center1_96.60.pt'

num_class = 2

alpha = 1.0
lr_cent = 0.5

# Training option
learning_rate = 1e-5
momentum_ = 0.90
total_epoch = 200
optimizer_type='radam' #'sgd', 'adam', 'radam'

## Original Dataset
with open('./dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

train_dataset = dataset['train']
test_dataset = dataset['test']

# delete -> label == 1
tr_remain = np.where(train_dataset[:, -1] != 1)
te_remain = np.where(test_dataset[:, -1] != 1)
tr_delete = np.where(train_dataset[:, -1] == 1)
te_delete = np.where(test_dataset[:, -1] == 1)
tr_change = np.where(train_dataset[:, -1] == 2)
te_change = np.where(test_dataset[:, -1] == 2)

dataset['train'][tr_change, -1] = 1
dataset['test'][te_change, -1] = 1

tr_aux = torch.unsqueeze(torch.tensor(dataset['train'][tr_delete, :-1])[0], dim=1).float().cuda()
te_aux = torch.unsqueeze(torch.tensor(dataset['test'][te_delete, :-1])[0], dim=1).float().cuda()

tr_x = torch.tensor(dataset['train'][tr_remain, :-1]).float()[0]
tr_y = torch.tensor(dataset['train'][tr_remain,-1]).long()[0]
te_x = torch.tensor(dataset['test'][te_remain,:-1]).float()[0]
te_y = torch.tensor(dataset['test'][te_remain,-1]).long()[0]

## Dataset & DataLoder
class Meat_data(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        dat_x = torch.unsqueeze(self.data_x[index], dim=0)
        dat_y = self.data_y[index]
        return dat_x, torch.tensor(-1000), dat_y

    def __len__(self):
        return len(self.data_x)

train_dataset = Meat_data(tr_x, tr_y)
test_dataset = Meat_data(te_x, te_y)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size = 512)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size = len(test_dataset))

## Load Model
saved = torch.load(pt_name)
if model_name == 0:
    model = AlexNet1D(num_classes=num_class, result_emb=True)
    model.up_type('cls')
    model.load_state_dict(saved['model'])
    model = model.cuda()

center_loss = AD_CenterLoss(num_classes=num_class, feat_dim=num_class, use_gpu=True)

soft_loss = nn.CrossEntropyLoss()
params = list(model.parameters()) + list(center_loss.parameters())
criterion = [soft_loss, center_loss]

# Optimizer
if optimizer_type.lower() == 'sgd':
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum_, nesterov=True)
elif optimizer_type.lower() == 'adam':
    optimizer = optim.Adam(params, lr=learning_rate)
elif optimizer_type.lower() == 'radam':
    from utils.optimizer import RAdam
    optimizer = RAdam(params, lr=learning_rate)

## Training and Evaluation
option = {'gpu':True, 'type_name':'cls', 'print_epoch':20, 'num_class':num_class, 'reg':False, 'lambda1':0, 'lambda2':0,
          'alpha':alpha, 'lr_cent':lr_cent, 'lr':learning_rate}

for epoch in tqdm(range(total_epoch)):
    model = ad_train(model, tr_aux, optimizer, epoch, train_loader, option, criterion, scheduler=None)

    if (epoch+1) % option['print_epoch'] == 0:
        model, te_loss, te_acc = test(model, epoch, test_loader, option, criterion)

## Save the model
os.makedirs('./result2', exist_ok=True)
torch.save({'model':model.state_dict(), 'loss':center_loss.state_dict()}, './result2/model%d_%.2f.pt' %(model_name, te_acc))
