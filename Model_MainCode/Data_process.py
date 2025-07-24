import numpy as np
import scipy
from scipy.io import loadmat

from src.Configuration_05 import DIRECTORY_PREFIX

matdata = loadmat(DIRECTORY_PREFIX + 'DATA_(200d100s).mat')  # 读取Mat文件
X = matdata['X']
Y_i = matdata['Y_i'][0]
Y_d = matdata['Y_d'][0]
selected_X = []
idx = []
num = 100 * 16
flag = False
original_X = X[list(range(0 * num, 1 * num))]


for d in [1, 2, 3, 4, 5, 6, 7]:
    if flag:
        new_X = X[list(range((d - 1) * num, d * num))] + original_X
        selected_X = np.vstack((selected_X, new_X))
    else:
        selected_X = X[list(range((d - 1) * num, d * num))] + original_X
        flag = True
    print(f"shape={selected_X.shape}")


print(f"shape={selected_X.shape},{Y_i.shape},{Y_d.shape}")

scipy.io.savemat(DIRECTORY_PREFIX + 'DATA_(200d100s).mat', {'X': selected_X, 'Y_i': Y_i,'Y_d':Y_d})


