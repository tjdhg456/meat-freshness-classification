import subprocess
import numpy as np
from multiprocessing import Process
import os
import torch

# Seed
torch.manual_seed(41)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(41)

# Basic Python Environment
python = '/home/lilka/anaconda3/envs/nlp_/bin/python'
folder = 'log_0131'

# Hyperparameter Candidate
gpu = [0,1,2,3,4,5,6]

num_size = 700
lr = np.random.choice([5e-2, 1e-2, 1e-3, 5e-4, 1e-5], size=num_size, replace=True)
lr_cent = np.random.choice([1e-1, 5e-2, 1e-2, 5e-3, 1e-4], size=num_size, replace=True)

lr2 = np.random.choice([5e-2, 1e-2, 1e-3, 5e-4, 1e-5], size=num_size, replace=True)
alpha = np.random.choice([0.3, 0.6, 0.9], size=num_size, replace=True)
aux = np.random.choice(['true', 'false'], size=num_size, replace=True)

# Combinations
comb = []
for (lr_, lr_cent_, lr2_, alpha_, aux_) in zip(lr, lr_cent, lr2, alpha, aux):
    comb += [(lr_, lr_cent_, lr2_, alpha_, aux_)]
arr = np.array_split(comb, len(gpu))
arr_dict = {}
for ix in range(len(gpu)):
    arr_dict[ix] = arr[ix]

# Training
def tr_gpu(comb, ix):
    comb = comb[ix]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(ix)
    for i, comb_ix in enumerate(comb):
        print('GPU %d : %d times' %(ix, i))
        script = '%s main.py --print_test False --gpu %d --lr %.5f --lr_cent %.3f --lr2 %.5f --alpha %.2f --log_folder %s --aux %s' %(python, gpu[ix], float(comb_ix[0]),
                                                                                                          float(comb_ix[1]), float(comb_ix[2]), float(comb_ix[3]),
                                                                                                          folder, str(comb_ix[4]))
        subprocess.call(script, shell=True)

for ix in range(len(gpu)):
    exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' %(ix, ix))

for ix in range(len(gpu)):
    exec('thread%d.start()' %ix)
