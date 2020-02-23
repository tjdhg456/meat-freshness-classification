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
# python = '/home/gyuri/anaconda3/envs/meat/bin/python'

folder = '../result/0223_final_late'
print(folder)

# Hyperparameter Candidate
gpu = [0,1,2,3,4,5]

# Condition
fusion = ['mid','late']
train_rule = ['resample', 'reweight', 'none']
sampler_type = ['SMOTE']
loss = ['focal', 'ce', 'ldam']
comb_list = []
ix = 0
num_per_gpu = 10

for f in fusion:
    for t in train_rule:
        for s in sampler_type:
            for l in loss:
                comb_list.append([f,t,s,l, ix])
                ix += 1

comb_list = comb_list * num_per_gpu

arr = np.array_split(comb_list, len(gpu))
arr_dict = {}
for ix in range(len(gpu)):
    arr_dict[ix] = arr[ix]

# Training
def tr_gpu(comb, ix):
    comb = comb[ix]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(ix)
    for i, comb_ix in enumerate(comb):
        folder_case = os.path.join(folder, str(comb_ix[4]))
        print('GPU %d : %d times' %(ix, i))
        script = '%s main.py --epoch_num 801 --print_test False \
         --gpu %d --fusion %s --train_rule %s --sampler_type %s --loss %s --log_folder %s' %(python, gpu[ix], str(comb_ix[0]),
                                                                                             str(comb_ix[1]), str(comb_ix[2]),
                                                                                             str(comb_ix[3]), folder_case)
        subprocess.call(script, shell=True)

for ix in range(len(gpu)):
    exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' %(ix, ix))

for ix in range(len(gpu)):
    exec('thread%d.start()' %ix)

