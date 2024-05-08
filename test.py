import LoadData as ld
import numpy as np

x = np.array([[1], [2], [3]])

y = np.array([4, 5, 6])

x = np.c_[x, y]
print(x.shape)