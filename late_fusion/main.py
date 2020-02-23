import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from module.model import AlexNet1D
from module.loss import TripletCenterLoss, CenterLoss, LDAMLoss, FocalLoss
import module.optimizer
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from utils.sampler import ImbalancedDatasetSampler, Sampler
from utils.trainer import train, test, ad_train
from utils.logging import log
from utils.earlystop import EarlyStopping
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import os
import argparse
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms
import warnings

## ArgParse
arg = argparse.ArgumentParser()
arg.add_argument('--gpu', default='3', type=str)
arg.add_argument('--save_model', default=True, type=lambda x: (str(x).lower() == 'true'))
arg.add_argument('--print_test', default=True, type=lambda x: (str(x).lower() == 'true'))
arg.add_argument('--print_epoch', default=40, type=int)
arg.add_argument('--log_folder', default='./imp', type=str)

arg.add_argument('--epoch_num', default=801, type=int)
arg.add_argument('--lr', default=1e-4, type=float)

arg.add_argument('--fusion', default='late', type=str)
arg.add_argument('--train_rule', default='resample', type=str) # None, Resample, Reweight, DRW
arg.add_argument('--sampler_type', default='smote', type=str) # binomial, up, down, SMOTE, border
arg.add_argument('--normalize', default='select', type=str)

arg.add_argument('--model', default='alexnet', type=str) # alexnet, vgg_11 ~~ , resnet_13 ~~
arg.add_argument('--loss', default='focal', type=str) # ce, ldam, focal

arg.add_argument('--LR_schedule', default=True, type=lambda x: (str(x).lower() == 'true'))
arg.add_argument('--earlystop', default=True, type=lambda x: (str(x).lower() == 'true'))

args = arg.parse_args()

if args.train_rule.lower() != 'resample':
    args.sampler_type = None

## GPU option
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
gpu = True

## Save Path
save_folder = './weight'
if os.path.isdir(save_folder) == False:
    exp_num = 0
else:
    exp_name = os.listdir(save_folder)
    if exp_name == []:
        exp_num = 0
    else:
        exp_num = sorted([int(name.lstrip('exp')) for name in exp_name])[-1] + 1

save_folder = os.path.join(save_folder, str(exp_num))
os.makedirs(save_folder, exist_ok=True)

## Load the raw spectrum data
'''
Data Path: './Data_pre/Meat_data.pkl'
Data Format : [(day1, s1, [ Wavelength ], [ spectrum ], [met, oxy, deoxy, sulf], pH), (day1, s2, [ ~~ ], [ ~~ ], ...
Data Type : text files, excel --> npz
Data Length : 2574 (78개 샘플 x 33 days)
'''

data_name = '../met_fusion/dataset_cut.pkl'

# Load the data
with open(data_name, 'rb') as f:
    data_dict = pickle.load(f)

tr_x = data_dict['tr_x']
tr_y = data_dict['tr_y']
tr_aux = data_dict['tr_aux']

te_x = data_dict['te_x']
te_y = data_dict['te_y']
te_aux = data_dict['te_aux']

## Define the Datset, DataLoader
class MeatData(Dataset):
    def __init__(self, list_x, list_y, list_aux, transform_x, transform_aux=None):
        self.data = list_x
        self.label = list_y
        self.aux = list_aux
        self.transform_x = transform_x
        self.transform_aux = transform_aux

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.from_numpy(self.data[index,:]).view([1,1,-1]).float()
        aux = torch.from_numpy(self.aux[index,:]).view([4,1,-1]).float()

        if self.transform_x is not None:
            x = self.transform_x(x)
        if self.transform_aux is not None:
            aux = self.transform_aux(aux)
        x = torch.squeeze(x, dim=1)
        aux = torch.squeeze(aux, dim=1)

        y = torch.from_numpy(self.label[[index]]).long()
        return x, aux, y

## Transform (normalization)
tr_m, tr_s = np.mean(tr_x), np.std(tr_x)
aux_m, aux_s = np.mean(tr_aux, axis=0), np.std(tr_aux, axis=0)

if args.normalize == 'all':
    transform_x = transforms.Normalize([tr_m], [tr_s])
    transform_aux = transforms.Normalize(aux_m, aux_s)
elif args.normalize =='select':
    transform_x = transforms.Normalize([tr_m], [tr_s])
    if args.fusion == 'early':
        transform_aux = transforms.Normalize(aux_m, aux_s)
    else:
        transform_aux = None
else:
    transform_x = None
    transform_aux = None

## Load the model
model_name = args.model
if 'alexnet' in model_name:
    out_channel = 512
    model_f = AlexNet1D
    option={}
else:
    raise('choose right model_name')

# Considering the fusion
model = model_f(in_channel=1, train_rule=args.fusion)
if gpu == True:
    model = model.cuda()

# Optimizer
optim_model = module.optimizer.RAdam(model.parameters(), lr=args.lr)

# LR Scheduler
if args.LR_schedule == True:
    scheduler = StepLR(optim_model, step_size=400, gamma=0.5)

# Early stopping
if args.earlystop == True:
    early = EarlyStopping(patience=5)

## Train and Evaluation
for epoch in range(args.epoch_num):

    ## Dataset and DataLoader definition :: Solving Imabalance Problems (Resample and Reweight)
    cls_num_list = []
    unique = sorted(np.unique(tr_y))
    for u in unique:
        cls_num_list.append(sum(tr_y == u))

    # Resample
    if args.train_rule.lower() == 'resample':
        per_cls_weights = None

        if args.sampler_type.lower() == 'smote':
            smote = SMOTE()
            train_x, train_y = smote.fit_resample(tr_x, tr_y)
            train_aux, _ = smote.fit_resample(tr_aux, tr_y)
        elif args.sampler_type.lower() == 'border':
            smote = BorderlineSMOTE()
            train_x, train_y = smote.fit_resample(tr_x, tr_y)
            train_aux, _ = smote.fit_resample(tr_aux, tr_y)
        else:
            train_x = tr_x
            train_y = tr_y
            train_aux = tr_aux

        train_set = MeatData(train_x, train_y, train_aux, transform_x, transform_aux)
        test_set = MeatData(te_x, te_y, te_aux, transform_x, transform_aux)

        if args.sampler_type.lower() == 'binomial':
            sampler = ImbalancedDatasetSampler(train_set)
        elif args.sampler_type.lower() == 'down':
            sampler = Sampler(train_set, type='under')
        elif args.sampler_type.lower() == 'up':
            sampler = Sampler(train_set, type='over')
        else:
            sampler = None

    else:
        train_set = MeatData(tr_x, tr_y, tr_aux, transform_x, transform_aux)
        test_set = MeatData(te_x, te_y, te_aux, transform_x, transform_aux)
        sampler = None

        # re-weighting
        if args.train_rule.lower() == 'reweight':
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            if gpu == True:
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            else:
                per_cls_weights = torch.FloatTensor(per_cls_weights)

        # re-weighting + DRW schedule
        elif args.train_rule.lower() == 'drw':
            train_sampler = None

            if epoch < 160:
                idx = 0
            else:
                idx = 1

            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            if gpu == True:
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            else:
                per_cls_weights = torch.FloatTensor(per_cls_weights)

        # None
        else:
            per_cls_weights = None

    train_loader = DataLoader(train_set, sampler=sampler, batch_size=512)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=len(test_set))

    # Loss
    if args.loss.lower() == 'ce':
        loss_cls = nn.CrossEntropyLoss(weight=per_cls_weights).cuda()
    elif args.loss.lower() == 'ldam':
        loss_cls = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda()
    elif args.loss.lower() == 'focal':
        loss_cls = FocalLoss(weight=per_cls_weights, gamma=1).cuda()

    # Training
    model = train(epoch, model, optim_model, loss_cls, train_loader, gpu, args)

    if args.LR_schedule == True:
        scheduler.step()

    # Save the stage0 model
    if epoch % args.print_epoch == 0:
        result = test(epoch, model, loss_cls, test_loader, gpu, args)

        if args.earlystop == True:
            # early(result['loss'], model, result)
            early(-result['f1']['macro'], model, result)
            if early.early_stop == True:
                break

        if args.print_test == True:
            print('Epoch : %d, Test Acc : %2.2f, Test Loss : %.2f' % (epoch, result['acc'], result['loss']))
            print(result['confusion'])
            print(result['f1'])

# Logging
model = early.model
result = early.result

result_log = {'acc':result['acc'], 'F1_Macro':result['f1']['macro'], 'confusion':str(result['confusion'])}
condition = {'epoch':epoch, 'lr':args.lr, 'fusion':args.fusion,
             'train_rule': args.train_rule, 'sampler_type' : args.sampler_type,
             'normalize': args.normalize, 'loss':args.loss}

log_option = {'num_class' : 3, 'top_k' : 10}
log(args.log_folder, condition, result_log, log_option, target='F1_Macro', gpu_num=None)

if args.save_model == True:
    save_folder = args.log_folder
    torch.save({'model':model.state_dict()},
               os.path.join(save_folder, 'best_model.pt'))

