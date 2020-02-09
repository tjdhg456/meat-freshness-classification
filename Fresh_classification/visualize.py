import numpy as np
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from module.model import AlexNet1D, classifier
import os
import argparse
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import seaborn as sns

## ArgParse
arg = argparse.ArgumentParser()
arg.add_argument('--gpu', default='2', type=str)
arg.add_argument('--model', default='alexnet', type=str)

arg.add_argument('--stage', default=0, type=int)
arg.add_argument('--exp_num', default=43, type=int)
arg.add_argument('--center', default=True, type=lambda x: (str(x).lower() == 'true'))
args = arg.parse_args()

## GPU option
if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    gpu = True

else:
    gpu = False

## option
save_folder = os.path.join('./weight', 'stage%d' %args.stage, '%d' %args.exp_num)
save_list = os.listdir(save_folder)
save_list = sorted(save_list, key=lambda x: int(x.lstrip('epoch').rstrip('.pt')))

## Load the raw spectrum data
# Load the data
with open('./dataset.pkl', 'rb') as f:
    data_dict = pickle.load(f)

tr_x = torch.unsqueeze(torch.from_numpy(data_dict['tr_x']), dim=1).float()
tr_y = torch.from_numpy(data_dict['tr_y']).long()
te_x = torch.unsqueeze(torch.from_numpy(data_dict['te_x']), dim=1).float()
te_y = torch.from_numpy(data_dict['te_y']).long()

if gpu == True:
    tr_x, te_x = tr_x.cuda(), te_x.cuda()

## Load the model
os.makedirs('./result_fig/%d' %args.exp_num, exist_ok=True)

for save in save_list:
    weight = os.path.join(save_folder, save)
    epoch = save.replace('.pt','').replace('epoch', '')

    train_folder = os.path.join('./result_fig', str(args.exp_num), 'train')
    test_folder = os.path.join('./result_fig', str(args.exp_num), 'test')

    plot_train_title = os.path.join(train_folder, save.replace('.pt','.png').replace('epoch', ''))
    plot_test_title = os.path.join(test_folder, save.replace('.pt','.png').replace('epoch', ''))

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # TODO: update the model (resnet, alexnet, vgg)
    if args.stage == 0:
        # Model
        if args.model.lower() == 'alexnet':
            emb = AlexNet1D()
            model = classifier(emb, 512, 2, gpu=gpu)
            model.load_state_dict(torch.load(weight)['model'])
            model.eval()

        # Center
        c0, c2 = torch.load(weight)['center']['centers'][0], torch.load(weight)['center']['centers'][1]
        c0, c2 = c0.cpu().detach().numpy(), c2.cpu().detach().numpy()

        try:
            c1 = torch.load(weight)['center']['aux_centers'][0]
            c1 = c1.cpu().detach().numpy()
            center_data = np.concatenate([np.concatenate([[c0], [c1], [c2]], axis=0), np.array([[0], [1], [2]])], axis=1)
        except:
            center_data = np.concatenate([np.concatenate([[c0], [c2]], axis=0), np.array([[0], [2]])], axis=1)


    if gpu == True:
        model = model.cuda()

    ## Get the value
    i = np.where(tr_y == 0)[0]
    j = np.where(tr_y == 1)[0]
    k = np.where(tr_y == 2)[0]
    tr_label = np.zeros(len(tr_y)).astype('str')
    tr_label[i] = 'fresh'
    tr_label[j] = 'normal'
    tr_label[k] = 'decay'

    i = np.where(te_y == 0)[0]
    j = np.where(te_y == 1)[0]
    k = np.where(te_y == 2)[0]
    te_label = np.zeros(len(te_y)).astype('str')
    te_label[i] = 'fresh'
    te_label[j] = 'normal'
    te_label[k] = 'decay'

    # Embedding
    with torch.no_grad():
        tr_out = model(tr_x).cpu().detach().numpy()
        te_out = model(te_x).cpu().detach().numpy()

    plt.figure(1)
    sns.scatterplot(tr_out[:,0], tr_out[:,1], hue=tr_label, legend='full', alpha=0.2, palette={'fresh':'blue','normal':'green','decay':'red'})
    if args.center == True:
        sns.scatterplot(center_data[:, 0], center_data[:, 1], hue=center_data[:,2], s=200, marker='+', palette={0:'blue',1:'green',2:'red'}, legend=False)
    plt.legend(title='category')
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    plt.title('epoch %s' %epoch)
    plt.savefig(plot_train_title)
    plt.close(1)

    plt.figure(2)
    sns.scatterplot(te_out[:,0], te_out[:,1], hue=te_label, legend='full', alpha=0.2, palette={'fresh':'blue','normal':'green','decay':'red'})
    if args.center == True:
        sns.scatterplot(center_data[:, 0], center_data[:, 1], hue=center_data[:,2], s=200, marker='+', palette={0:'blue',1:'green',2:'red'}, legend=False)
    plt.legend(title='category')
    # plt.xlim(-1,1)
    # plt.ylim(-1,1)
    plt.title('epoch %s' %epoch)
    plt.savefig(plot_test_title)
    plt.close(2)
    del model