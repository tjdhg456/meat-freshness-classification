import subprocess
import numpy as np
from multiprocessing import Process
import os
import torch

# # Seed
# torch.manual_seed(41)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(41)

# Basic Python Environment
python = '/home/lilka/anaconda3/envs/nlp_/bin/python'
norm = 'select'
folder = '0221_result' + norm
print(folder)

# Hyperparameter Candidate
gpu = [0,1,2,3,4,5]

lr = [1e-3] * 300 + [1e-4] * 300
fusion = (['none'] * 100 + ['early'] * 100 + ['pred'] * 100) * 2
comb = []
for (lr_, fusion_) in zip(lr, fusion):
    comb += [(lr_, fusion_, norm)]

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
        script = '%s main.py --epoch_num 801 --print_test False --gpu %d --lr %.5f --fusion %s --normalize %s --log_folder %s' %(python, gpu[ix], float(comb_ix[0]),
                                                                                                 str(comb_ix[1]), str(comb_ix[2]), folder)
        subprocess.call(script, shell=True)

for ix in range(len(gpu)):
    exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' %(ix, ix))

for ix in range(len(gpu)):
    exec('thread%d.start()' %ix)

