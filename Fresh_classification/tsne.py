import numpy as np
import torch
from MulticoreTSNE import MulticoreTSNE as TSNE
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from module.model import AlexNet1D
import os
# Option
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

model_name = 0
original = False
center = True

if original == True:
    if center == True:
        pt_name = './result/model0_epoch500_center1_97.01.pt' # with center loss, 500 epoch
        pt_name = './result/model0_epoch700_center1_96.60.pt' # with center loss, 700 epoch
    else:
        pt_name = './result/model0_epoch500_center0_96.73.pt' # without center loss, 500 epoch
else:
    center = True
    pt_name = './result2/model0_97.41.pt' # with center loss, additional 200 epoch with imbalance center loss

num_class = 2

# Original Dataset
with open('./dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

tr_x = torch.unsqueeze(torch.tensor(dataset['train'][:, :-1]).float(), dim=1).cuda()
tr_y = dataset['train'][:,-1]
te_x = torch.unsqueeze(torch.tensor(dataset['test'][:,:-1]).float(), dim=1).cuda()
te_y = dataset['test'][:,-1]

# Load Model
if model_name == 0:
    model = AlexNet1D(num_classes=num_class, result_emb=True)
    model.up_type('cls')
    model.load_state_dict(torch.load(pt_name)['model'])
    model = model.cuda()

if original == False:
    center_loss = torch.load(pt_name)['loss']
    c0, c2, c1 = center_loss['centers'][0], center_loss['centers'][1], center_loss['aux_centers'][0]
    c0, c2, c1 = c0.cpu().detach().numpy().reshape(1,-1), c2.cpu().detach().numpy().reshape(1,-1), c1.cpu().detach().numpy().reshape(1,-1)
    center_data = np.concatenate([np.concatenate([c0,c1,c2], axis=0), np.array([[0],[1],[2]])], axis=1)
else:
    if center == True:
        center_loss = torch.load(pt_name)['loss']
        c0, c2 = center_loss['centers'][0], center_loss['centers'][1]
        c0, c2 = c0.cpu().detach().numpy().reshape(1,-1), c2.cpu().detach().numpy().reshape(1,-1)
        center_data = np.concatenate([np.concatenate([c0, c2], axis=0), np.array([[0],[2]])], axis=1)


tr_out, tr_mid = model(tr_x)
te_out, te_mid = model(te_x)

## TSNE Embedding
# tr_emb = tr_mid.cpu().detach().numpy()
# te_emb = te_mid.cpu().detach().numpy()
tr_emb = tr_out.cpu().detach().numpy()
te_emb = te_out.cpu().detach().numpy()

tsne_model = TSNE(n_jobs=4, n_components=2, perplexity=60.0, learning_rate=100)
# tsne_tr = tsne_model.fit_transform(tr_emb)
# tsne_te = tsne_model.fit_transform(te_emb)
tsne_tr = tr_emb
tsne_te = te_emb

# Visualization with TSNE
plt.figure(1)
sns.scatterplot(tsne_tr[:,0], tsne_tr[:,1], hue=tr_y, legend='full', alpha=0.5)
if center == True:
    sns.scatterplot(center_data[:,0], center_data[:,1], hue=center_data[:,2], legend='full', s=200, marker='+')
plt.title('tSNE : train dataset with embedded')
plt.show()

plt.figure(2)
sns.scatterplot(tsne_te[:,0], tsne_te[:,1], hue=te_y, legend='full', alpha=0.5)
plt.title('tSNE : test dataset with embedded')
plt.show()
