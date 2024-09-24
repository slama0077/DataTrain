import numpy as np
from utils import LoadData as ld
from utils import Welchh
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


pandaArray = pandas.Series(bandpower_list[0:21], index = ["FP1", "FP2", "F7", "F8", "F3", "F4", "FZ", "T3", "T4", "C3", "C4", "CZ", "P3", "P4", "PZ", "T5", "T6", "O1", "O2", "A1", "A2"])

fig = topoplot(pandaArray, n_colors = 200, title = "Relative Deltapower")
plt.figure(fig)
plt.show()
