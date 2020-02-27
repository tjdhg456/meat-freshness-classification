import pandas as pd
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import os

## Visualize low vs high spectrum
low_name = '../Data_pre/0213_low_MeatData.pkl'
with open(low_name, 'rb') as f:
    dataset_l = pickle.load(f)

high_name = '../Data_pre/MeatData_0213.pkl'
with open(high_name, 'rb') as f:
    dataset_h = pickle.load(f)

day = 'day14'
sample = 's5'

l_index = np.asarray([[l[0], l[1]] for l in dataset_l])
h_index = np.asarray([[h[0], h[1]] for h in dataset_h])

i = set(np.where(l_index[:,0] == day)[0].tolist())
c = set(np.where(l_index[:,1] == sample)[0].tolist())
l_ix = list(i.intersection(c))[0]

i = set(np.where(h_index[:,0] == day)[0].tolist())
c = set(np.where(h_index[:,1] == sample)[0].tolist())
h_ix = list(i.intersection(c))[0]

## Data
l_spec = dataset_l[l_ix][2][201:1720]
l_data = dataset_l[l_ix][3][201:1720]

h_spec = dataset_h[h_ix][2][30:-1487]
h_data = dataset_h[h_ix][3][30:-1487]

# Drawing twin-yaxis graph
fig = host_subplot(111)
par = fig.twinx()

fig.set_title('Spectrum at %s-%s' %(day, sample))
fig.set_xlabel('Wavelength(nm)')
fig.set_ylabel('Reflectance', fontsize=13)
par.set_ylabel('Reflectance', fontsize=13)

# fig.set_title('Meat' + str(num))
p0, = fig.plot(l_spec, l_data, 'r', label="low SNR", linewidth=3)
fig.set_ylim(0, 1.2)
fig.grid(False)

p1, = par.plot(h_spec, h_data, 'b', label="high SNR", linewidth=3)
par.set_ylim(0, 1.2)
par.grid(False)

# Legend
leg = plt.legend(loc='upper right')
leg.texts[0].set_color(p0.get_color())
leg.texts[1].set_color(p1.get_color())


# Save the image
save_folder = './plot'
os.makedirs(save_folder, exist_ok=True)
save_name = os.path.join(save_folder, ('Meat_%s_%s' %(day, sample)))
plt.savefig(save_name)
plt.close()

exit()

## Result Data
result = pd.read_excel('./result_boxplot.xlsx')


## Boxplot
bplot = sns.boxplot(x="fusion", y="f1", hue="loss",
                 data=result, palette="Set3")

bplot.axes.set_title("f1 results", fontsize=12)
bplot.set_xlabel("fusion method", fontsize=10)
bplot.set_ylabel("f1 score", fontsize=10)
bplot.tick_params(labelsize=8)


# output file name
plot_file_name = "boxplot_f1.png"

# save as jpeg
bplot.figure.savefig(plot_file_name,
                     format='png', dpi=120)