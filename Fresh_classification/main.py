import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from module.model import AlexNet1D, ResNet1D, VGG1D, classifier
from module.loss import CenterLoss, AD_CenterLoss, TripletCenterLoss
import module.optimizer
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from utils.sampler import ImbalancedDatasetSampler, Sampler
from utils.trainer import train, test, ad_train
from utils.logging import log
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import argparse
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms
import warnings

warnings.filterwarnings("ignore")

# Seed
torch.manual_seed(43)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(43)

## ArgParse
arg = argparse.ArgumentParser()
arg.add_argument('--gpu', default='0', type=str)
arg.add_argument('--save_data', default=False, type=lambda x: (str(x).lower() == 'true'))
arg.add_argument('--save_model', default=False, type=lambda x: (str(x).lower() == 'true'))
arg.add_argument('--print_test', default=True, type=lambda x: (str(x).lower() == 'true'))
arg.add_argument('--print_epoch', default=40, type=int)
arg.add_argument('--log_folder', default='./imp', type=str)
arg.add_argument('--aux', default=False, type=lambda x: (str(x).lower() == 'true'))
arg.add_argument('--model', default='alexnet', type=str)    # vgg11, 13, 16, 19 + _bn, resnet18, 34, 50, alexnet

arg.add_argument('--lr', default=1e-4, type=float)
arg.add_argument('--lr_cent', default=1e-3, type=float)
arg.add_argument('--lr2', default=1e-3, type=float)

arg.add_argument('--batch', default=512, type=int)
arg.add_argument('--epoch_num1', default=121, type=int)
arg.add_argument('--epoch_num2', default=301, type=int)

arg.add_argument('--stage', default=0, type=int)
arg.add_argument('--load_exp', default=45, type=int)
arg.add_argument('--load_epoch', default=200, type=int)
arg.add_argument('--alpha', default=0.0, type=float)
arg.add_argument('--scheduler', default=False, type=lambda x: (str(x).lower() == 'true'))

arg.add_argument('--sampler_type', default=4, type=int)

args = arg.parse_args()

## GPU option
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu = True
else:
    gpu = False

## option
save_folder = './weight'
os.makedirs(os.path.join(save_folder, 'stage0'), exist_ok=True)
os.makedirs(os.path.join(save_folder, 'stage1'), exist_ok=True)

exp_name = os.listdir(os.path.join(save_folder, 'stage0'))
if exp_name == []:
    exp_num = 0
else:
    exp_num = sorted([int(name.lstrip('exp')) for name in exp_name])[-1] + 1

if args.save_model == True:
    os.makedirs(os.path.join(save_folder, 'stage0', str(exp_num)), exist_ok=True)
    os.makedirs(os.path.join(save_folder, 'stage1', str(exp_num)), exist_ok=True)

if args.print_test == True:
    print(exp_num)

## Load the raw spectrum data
'''
Data Path: './Data_pre/Meat_data.pkl'
Data Format : [(day1, s1, [ Wavelength ], [ spectrum ], pH, met), (day1, s2, [ ~~ ], [ ~~ ], pH, met), ..., (day33, s78, [ ~~ ], [ ~~ ], pH, met)
Data Type : text files, excel --> npz
Data Length : 2574 (78개 샘플 x 33 days)
'''
if args.save_data == True:
    data_meat = np.load('../data/Meat_data_Reflectance.npy', allow_pickle=True)

    # Total Data
    spec = data_meat[0][2]
    x_data, y_data = [], []
    for data_ in data_meat:
        x = data_[3]
        y_raw = data_[4]
        if y_raw < 6.0:
            y = 0
        elif y_raw < 6.3:
            y = 1
        else:
            y = 2

        x_data += [x]
        y_data += [y]

    x_data, y_data = np.asarray(x_data), np.asarray(y_data)
    tr_x, te_x, tr_y, te_y = train_test_split(x_data, y_data, test_size=0.3, random_state=41, stratify=y_data)

    data_dict = {'tr_x':tr_x, 'tr_y':tr_y, 'te_x':te_x, 'te_y':te_y}

    # Save the data
    with open('./dataset.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

else:
    # Load the data
    with open('./dataset.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    tr_x = data_dict['tr_x']
    tr_y = data_dict['tr_y']
    te_x = data_dict['te_x']
    te_y = data_dict['te_y']

## Define the Datset, DataLoader
class MeatData(Dataset):
    def __init__(self, list_x, list_y, stage=0, transform=None):
        self.data = list_x
        self.label = list_y

        if stage == 0:
            ix = np.where(self.label == 2)[0]
            self.label[ix] = 1

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.unsqueeze(torch.from_numpy(self.data[index,:]), dim=0).float()
        y = torch.from_numpy(self.label[[index]]).long()

        if self.transform is not None:
            x = self.transform(x)

        return x, y

## stage0 : Using only 0, 2 label
# Load the train/test Dataset/DataLoader for stage0 (delete label 1)
# TODO: update the mean and standard deviation of spectrum
# trans_norm = transforms.Normalize([],[])
trans_norm = None

tr_ix = np.where(tr_y != 1)[0]
tr_i = np.where(tr_y == 1)[0]
te_ix = np.where(te_y != 1)[0]

train_set1 = MeatData(tr_x[tr_ix, :], tr_y[tr_ix], stage=0, transform=trans_norm)
test_set1 = MeatData(te_x[te_ix, :], te_y[te_ix], stage=0, transform=trans_norm)

if args.aux:
    train_aux = torch.unsqueeze(torch.from_numpy(tr_x[tr_i,:]), dim=1).float()
    if gpu == True:
        train_aux = train_aux.cuda()
else:
    train_aux = None

train_loader1 = DataLoader(train_set1, shuffle=True, batch_size=args.batch)
test_loader1 = DataLoader(test_set1, shuffle=False, batch_size=len(test_set1))

## Load the saved weight
load_pt = os.path.join(str(args.load_exp), 'epoch%03d.pt' %args.load_epoch)

## Load the model
# TODO: update the model (resnet, alexnet, vgg)
if args.model.lower() == 'alexnet':
    emb = AlexNet1D()
    in_channel = 512
elif 'resnet' in args.model.lower():
    emb = ResNet1D(args.model)
    in_channel = int(512 * emb.expansion)
elif 'vgg' in args.model.lower():
    emb = VGG1D(args.model)
    in_channel = 2048

if gpu == True:
    emb = emb.cuda()

model = classifier(emb, in_channel, 2, gpu)

# Training
option = {'gpu':gpu, 'alpha':args.alpha, 'print_epoch':args.print_epoch, 'aux':train_aux, 'print':args.print_test}

## Define the loss and optimizer
loss_cls = nn.CrossEntropyLoss()

if args.alpha != 0.0:
    if args.aux:
        loss_center = TripletCenterLoss(margin=0.5, num_classes=3, use_gpu=gpu)
    else:
        loss_center = TripletCenterLoss(margin=0.5, num_classes=2, use_gpu=gpu)
else:
    loss_center = CenterLoss(num_classes=2, feat_dim=2, use_gpu=gpu)

optim_model = module.optimizer.RAdam(model.parameters(), lr=args.lr)
optim_center = module.optimizer.RAdam(loss_center.parameters(), lr=args.lr_cent)

optimizer = {'cls':optim_model, 'center':optim_center}
criterion = {'cls':loss_cls, 'center':loss_center}

# LR Scheduler
if args.scheduler == True:
    scheduler_cls = StepLR(optimizer['cls'], step_size=100, gamma=0.5)
    scheduler_center = StepLR(optimizer['center'], step_size=100, gamma=0.1)

if args.stage == 0:
    # Train the stage0
    for epoch in range(args.epoch_num1):
        model = train(epoch, model, optimizer, criterion, train_loader1, option)
        # Save the stage0 model
        if epoch % option['print_epoch'] == 0:
            result = test(epoch, model, criterion, test_loader1, option, num_class=2)
            if args.print_test == True:
                print('Epoch : %d, Test Acc : %2.2f, Test Loss : %.2f' % (epoch, result['acc'], result['loss']))
                print(result['confusion'])
                print(result['f1'])

            if args.save_model == True:
                torch.save({'model':model.state_dict(),
                           'center':loss_center.state_dict()},
                           os.path.join(save_folder, 'stage0', str(exp_num), 'epoch%03d.pt' %epoch))

        if args.scheduler == True:
            scheduler_cls.step()
            scheduler_center.step()
else:
    weight1 = torch.load(os.path.join(save_folder, 'stage0', load_pt))
    model.load_state_dict(weight1['model'])

## stage1 : Using all labels
# Load the train/test Dataset/DataLoader for stage1
if args.sampler_type == 4:
    smote = SMOTE()
    tr_x, tr_y = smote.fit_resample(tr_x, tr_y)
    tr_y = tr_y.reshape(-1,1)
elif args.sampler_type == 5:
    smote = BorderlineSMOTE()
    tr_x, tr_y = smote.fit_resample(tr_x, tr_y)
    tr_y = tr_y.reshape(-1,1)

train_set = MeatData(tr_x, tr_y, stage=1, transform=trans_norm)
test_set = MeatData(te_x, te_y, stage=1, transform=trans_norm)

if args.sampler_type == 1:
    sampler = ImbalancedDatasetSampler(train_set)
elif args.sampler_type == 2:
    sampler = Sampler(train_set, type='under')
elif args.sampler_type == 3:
    sampler = Sampler(train_set, type='over')

if args.sampler_type in [1,2,3]:
    train_loader = DataLoader(train_set, sampler= sampler, batch_size = 512)
else:
    train_loader = DataLoader(train_set, shuffle=True, batch_size=512)

test_loader = DataLoader(test_set, shuffle=False, batch_size=len(test_set))

# Training stage1
option = {'lr':args.lr2, 'gpu':gpu, 'print_epoch':args.print_epoch, 'print':args.print_test}
model_emb = classifier(model.eval(), in_num=2, out_num=3, gpu=gpu, freeze=True, init_weights=False)

del train_set1, train_loader1, test_set1, test_loader1
del loss_cls, loss_center, optim_model, optim_center
loss_cls = nn.CrossEntropyLoss()
optim_model = module.optimizer.RAdam(model_emb.parameters(), lr=args.lr)
optimizer = {'cls':optim_model}
criterion = {'cls':loss_cls}

if args.stage < 2:
    for epoch in range(args.epoch_num2):
        model_emb = ad_train(epoch, model_emb, optimizer, criterion, train_loader, option)
        # Save the stage0 model
        if epoch % option['print_epoch'] == 0:
            result = test(epoch, model_emb, criterion, test_loader, option, num_class=3)
            if args.print_test:
                print('Epoch : %d, Test Acc : %2.2f, Test Loss : %.2f' % (epoch, result['acc'], result['loss']))
                print(result['confusion'])
                print(result['f1'])

            # Logging
            result_log = {'acc':result['acc'], 'F1_Macro':result['f1']['macro'], 'confusion':str(result['confusion'])}
            condition = {'epoch':epoch, 'lr':args.lr, 'lr_cent':args.lr_cent,
                         'lr2' : args.lr2, 'alpha':args.alpha, 'model':args.model,
                         'sampler_type':args.sampler_type, 'aux':args.aux}
            log_option = {'num_class' : 3, 'top_k' : 5}
            log(args.log_folder, condition, result_log, log_option, target='F1_Macro', gpu_num=args.gpu)

            if args.save_model == True:
                torch.save({'model':model_emb.state_dict()},
                           os.path.join(save_folder, 'stage1', str(exp_num), 'epoch%03d.pt' %epoch))

else:
    weight2 = torch.load(os.path.join(save_folder, 'stage1', load_pt))
    model_emb.load_state_dict(weight2['model'])

exit()