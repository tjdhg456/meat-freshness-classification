import numpy as np
import glob
import pandas as pd
import os, sys
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
from mpl_toolkits.axes_grid1 import host_subplot

## Convert Text files and Excels into npz
'''
Data Format : [(day1, s1, [ Wavelength ], [ spectrum ], pH, met), (day1, s2, [ ~~ ], [ ~~ ], pH, met), ..., (day33, s78, [ ~~ ], [ ~~ ], pH, met)
Data Type : text files, excel --> npz
Data Length : 2574 (78개 샘플 x 33 days)
'''
def data_merge():
    list_spectrum = glob.glob('./2mm/*/s*.txt')
    list_spectrum = [spec.replace('\\','/') for spec in list_spectrum]
    list_spectrum.sort(key=lambda x: int(x.split('/')[-1][1:-4]))

    data_ph = pd.read_excel('./label_pH_met.xlsx', sheet_name='ph')
    data_met = pd.read_excel('./label_pH_met.xlsx', sheet_name ='met')

    ## Save the Spectrum data as pickle
    data_zip = []
    for ix in range(len(list_spectrum)):
        file_name = list_spectrum[ix]
        sample_num = int(file_name.split('/')[-1][1:-4])
        day = int(file_name.split('/')[-2][3:])

        with open(list_spectrum[ix], 'r') as f:
            data_spectrum = f.readlines()

        ph = float(data_ph.loc[data_ph.Sample == sample_num, day])
        met = float(data_met.loc[data_met.Sample == sample_num, day])

        data_spectrum = data_spectrum[25 : -10]
        spec = [float(data.split('\t')[1]) for data in data_spectrum]
        ref = [float(data.split('\t')[0]) for data in data_spectrum]
        data_zip.append(('day%d' %day, 's%d' %sample_num, ref, spec, ph, met))

    np.save('./Meat_data.npy', data_zip)

    ## Save the label as pickle
    column_list = list(data_ph.columns)
    column_list.remove('Sample')

    sample = np.array(data_ph.index) + 1
    grade = {}

    for sample_ix in sample:
        met_ix = data_met.loc[data_met.Sample == sample_ix, column_list]
        ph_ix = data_ph.loc[data_ph.Sample == sample_ix, column_list]

        met_ix = np.array(met_ix).reshape(-1)
        ph_ix = np.array(ph_ix).reshape(-1)

        met_min = np.where((met_ix >= 0.2) & (met_ix < 0.4))
        met_mid = np.where(met_ix >= 0.4)
        ph_min = np.where((ph_ix >= 6.0) & (ph_ix < 6.3))
        ph_mid = np.where(ph_ix >= 6.3)

        index = [met_min, met_mid, ph_min, ph_mid]
        index_list = []
        for ind in index:
            if list(ind[0]) != []:
                if int(ind[0][0]) == 0:
                    index_list.append(1000)
                else:
                    index_list.append(int(ind[0][0])+1)
            else:
                index_list.append(1000)
        grade[str(sample_ix)] = index_list

    np.savez('./Meat_label.npz', **grade)


## Plot
def plotting():
    data_imp = np.load('./Meat_data.npy', allow_pickle=True)

    meat_num = 78 # The number of meat
    day_num = 33 # The number of day

    for num in range(1, meat_num+1):
        ph = [data_[4] for data_ in data_imp if (data_[1] == 's' + str(num))]
        met = [float('%.2f'%(data_[5] * 100)) for data_ in data_imp if (data_[1] == 's' + str(num))]

        ph_imp = np.array(ph)
        met_imp = np.array(met)

        # index of changing place
        met_min = np.where(met_imp >= 20)
        met_mid = np.where(met_imp >= 40)
        ph_min = np.where(ph_imp >= 6.0)
        ph_mid = np.where(ph_imp >= 6.3)

        index = [met_min, met_mid, ph_min, ph_mid]
        index_list = []
        for ind in index:
            if list(ind[0]) != []:
                if int(ind[0][0]) == 0:
                    index_list.append('')
                else:
                    index_list.append(int(ind[0][0]))
            else:
                index_list.append('')

        # Drawing twin-yaxis graph
        fig = host_subplot(111)
        par = fig.twinx()

        fig.set_xlabel('Day')
        fig.set_ylabel('Percentage of met-myoglobin(%)', fontsize=13)
        par.set_ylabel('pH', fontsize=13)

        fig.set_title('Meat' + str(num))
        p0,  = fig.plot(list(range(1, day_num+1)), met, 'rs-', label="met")
        fig.set_ylim(0, 100)
        fig.grid(False)

        p1, = par.plot(list(range(1, day_num+1)), ph, 'bs-', label="pH")
        par.set_ylim(5, 8)
        par.grid(False)

        # Draw the star for the changing point
        met_index = index_list[:2]
        ph_index = index_list[2:]
        for ix in range(2):
            if met_index[ix] != '':
                fig.plot((met_index[ix]+1), met[met_index[ix]], 'm*', ms=16)
            if ph_index[ix] != '':
                par.plot((ph_index[ix]+1), ph[ph_index[ix]], 'g*', ms=16)

        # Legend
        leg = plt.legend(loc='upper right')
        leg.texts[0].set_color(p0.get_color())
        leg.texts[1].set_color(p1.get_color())

        # Save the image
        save_folder = './plot/twin_ph_met'
        os.makedirs(save_folder, exist_ok=True)
        save_name = os.path.join(save_folder, ('Meat'+str(num)+'.png'))
        plt.savefig(save_name)
        plt.close()

## Implementation
# data_merge()

# plotting()
