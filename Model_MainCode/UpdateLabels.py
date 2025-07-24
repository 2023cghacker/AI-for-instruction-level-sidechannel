"""
    @Author : ling chen
    @Create : 2023/07
    @Last modify: 2023/07
    @Description: 给含寄存器信息的数据集加上第一层标签，大类标签
"""

import time
import numpy as np
import scipy
from scipy.io import loadmat
from src.Configuration_03 import DIRECTORY_PREFIX, INSTRUCTION_REGNAME, INSTRUCTION_LABEL

begint = time.time()

"""
1、导入测试集和训练集
"""
address = DIRECTORY_PREFIX + '1T_traintest_m(500d100s).mat'
matdata = loadmat(address)  # 读取Mat文件, 文件里有训练集和测试集
train_X = matdata['train_X']
train_Y = matdata['train_Y'][0]
test_X = matdata['test_X']
test_Y = matdata['test_Y'][0]
print("训练集输入规模", train_X.shape, "训练集标签规模", train_Y.shape)
print("测试集输入规模", test_X.shape, "测试集标签规模", test_Y.shape)


train_Y2 = train_Y / 10
train_Y2 = np.trunc(train_Y2).astype(int)  # 取整
print(train_Y2)
test_Y2 = test_Y / 10
test_Y2 = np.trunc(test_Y2).astype(int)  # 取整
print(test_Y2)

scipy.io.savemat(address,
                 {'train_X': train_X, 'train_Y': train_Y, 'train_Y2': train_Y2, 'test_X': test_X, 'test_Y': test_Y,
                  'test_Y2': test_Y2})
