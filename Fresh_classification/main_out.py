import numpy as np
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from module.model import AlexNet1D, EMB_cls
from module.trainer import train_out, test_out
from module.loss import AD_CenterLoss
import torch.nn as nn
import torch.optim as optim
import utils
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from utils.sampler import ImbalancedDatasetSampler, Sampler

# Option
np.random.seed(42)
torch.manual_seed(10)

model_name = 0
# pt_name = './result2_old/model0_97.41.pt'
pt_name = './result_old/model0_epoch500_center1_97.01.pt'

sampler_type = 4    # 1: Binomial, 2: Under, 3: Over, 4: SMOTE, 5: Borderline SMOTE

num_class = 2
out_class = 3


# Training option
learning_rate = 1e-3
momentum_ = 0.90
total_epoch = 600
optimizer_type='radam' #'sgd', 'adam', 'radam'

## Original Dataset
with open('./dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

train_dataset = dataset['train']
test_dataset = dataset['test']

# To numpy array (Train/Test Split)
tr_x = train_dataset[:, :-1]
tr_y = train_dataset[:,-1]
te_x = test_dataset[:,:-1]
te_y = test_dataset[:,-1]

if sampler_type == 4:
    smote = SMOTE()
    tr_x, tr_y = smote.fit_resample(tr_x, tr_y)
    tr_y = tr_y.reshape(-1,1)
elif sampler_type == 5:
    smote = BorderlineSMOTE()
    tr_x, tr_y = smote.fit_resample(tr_x, tr_y)
    tr_y = tr_y.reshape(-1,1)

# To torch tensor
tr_x = torch.from_numpy(tr_x).float()
tr_y = torch.from_numpy(tr_y).long()
te_x = torch.from_numpy(te_x).float()
te_y = torch.from_numpy(te_y).long()

## Dataset & DataLoder
class Meat_data(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        dat_x = torch.unsqueeze(self.data_x[index], dim=0)
        dat_y = self.data_y[index]
        return dat_x, dat_y

    def __len__(self):
        return len(self.data_x)

train_dataset = Meat_data(tr_x, tr_y)
test_dataset = Meat_data(te_x, te_y)

if sampler_type == 1:
    sampler = ImbalancedDatasetSampler(train_dataset)
elif sampler_type == 2:
    sampler = Sampler(train_dataset, type='under')
elif sampler_type == 3:
    sampler = Sampler(train_dataset, type='over')

if sampler_type in [1,2,3]:
    train_loader = DataLoader(train_dataset, sampler= sampler, batch_size = 512)
else:
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=512)

test_loader = DataLoader(test_dataset, shuffle=False, batch_size = len(test_dataset))

## Load Model
saved = torch.load(pt_name)
if model_name == 0:
    model_base = AlexNet1D(num_classes=num_class, result_emb=True)
    model_base.up_type('cls')
    model_base.load_state_dict(saved['model'])
    model_base = model_base.cuda().eval()

    model = EMB_cls(model=model_base, out_class=3)

soft_loss = nn.CrossEntropyLoss()

# Optimizer
params = model.parameters()
if optimizer_type.lower() == 'sgd':
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum_, nesterov=True)
elif optimizer_type.lower() == 'adam':
    optimizer = optim.Adam(params, lr=learning_rate)
elif optimizer_type.lower() == 'radam':
    from utils.optimizer import RAdam
    optimizer = RAdam(params, lr=learning_rate)

## Training and Evaluation
option = {'gpu':True, 'type_name':'cls', 'print_epoch':20, 'num_class':out_class, 'reg':False, 'lambda1':0, 'lambda2':0,
          'lr':learning_rate}

for epoch in tqdm(range(total_epoch)):
    model = train_out(model, optimizer, epoch, train_loader, option, soft_loss)

    if (epoch+1) % option['print_epoch'] == 0:
        model, te_acc = test_out(model, epoch, test_loader, option)

## Save the model
os.makedirs('./result3', exist_ok=True)
torch.save({'model':model.state_dict()}, './result3/model%d_sampler%d_%.2f.pt' %(model_name, sampler_type, te_acc))
