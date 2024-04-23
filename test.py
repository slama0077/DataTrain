import LoadData as ld
import numpy as np

raw_data_hand = ld.loadData()  
raw_data_hand = np.transpose(raw_data_hand)

print((raw_data_hand[:, 0:150]).shape)