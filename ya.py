import numpy as np
import LoadData as ld
import Welchh
from yasa import topoplot
import pandas
import matplotlib.pyplot as plt
raw_data = ld.loadData()
sf = 250
print(raw_data.shape)
bandpower_list = []
ch_names = []
for i in range(raw_data.shape[1]):
    channel_list= []
    for j in range(raw_data.shape[0]):
        channel_list.append(raw_data[j][i])
    
    bandpower = Welchh.bandpower_multitaper(np.array(channel_list), sf, [0,4])
    bandpower_list.append(bandpower)


pandaArray = pandas.Series(bandpower_list[0:7], index = ['F4', 'F3', 'C4', 'C3', 'P3', 'P4', 'Oz'], name = 'Values')
print(pandaArray)

# fig = topoplot(pandaArray, n_colors = 8, cmap = 'Reds', title = "Relative Deltapower")
# plt.figure(fig)
# plt.show()