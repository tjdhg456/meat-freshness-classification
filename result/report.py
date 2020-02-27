import os
import numpy as np
import argparse
import pandas as pd

## ArgParse
arg = argparse.ArgumentParser()
arg.add_argument('--save_folder', default='./0224_final', type=str)
arg.add_argument('--earlystop', default=True, type=lambda x: (str(x).lower() == 'true'))

args = arg.parse_args()

## Path
case_list = os.listdir(args.save_folder)
cond_list = []
for ix, case in enumerate(case_list):
    txt_file = os.path.join(args.save_folder, case, 'log.txt')
    with open(txt_file, 'r') as f:
        txt = f.readlines()
        txt = np.asarray(txt)

    # result
    result_ix = np.where(txt == '[Result]\n')[0] + 1
    result = [float(t.split(',')[1].split(':')[1].strip()) for t in txt[result_ix]]
    u, s = np.mean(result), np.std(result)

    # condition
    condition_ix = np.where(txt == '[Condition]\n')[0][0] + 1
    condition = txt[(condition_ix) : (condition_ix+3)]

    target_list = ['fusion', 'train_rule', 'sampler_type', 'loss']
    target_cond = [[c_.split(':')[0].strip(), c_.split(':')[1].strip()] for cond_i in condition for c_ in cond_i.split(',') if c_.split(':')[0].strip() in target_list]
    target_cond = np.asarray(target_cond)

    cond = target_cond[:, 1].tolist() + [u, s, case]

    cond_list.append(cond)

# Report
sheet = target_cond[:, 0].tolist() + ['mean', 'std', 'case_ix']
report = pd.DataFrame(np.asarray(cond_list), columns=sheet)
report = report.sort_values(by=['mean','fusion', 'train_rule', 'sampler_type', 'loss'], ascending=False)

save_name = args.save_folder + '.csv'
report.to_csv(save_name, index=False)


