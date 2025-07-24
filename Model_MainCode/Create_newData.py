import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import loadmat
from scipy.stats import kstest, shapiro, kurtosis, norm, skew
import src.Configuration_matplot
from src.Configuration_04 import DIRECTORY_PREFIX, Absolute_address

'''
导入原始数据集
'''
address = DIRECTORY_PREFIX + 'traintest_m(4ou750d200s).mat'
newdataname = DIRECTORY_PREFIX + 'traintest_m(4ou750d200s)_expand.mat'
matdata = loadmat(address)  # 读取Mat文件, 文件里有训练集和测试集
train_X = matdata['train_X']
train_Y = matdata['train_Y'][0]
test_X = matdata['test_X']
test_Y = matdata['test_Y'][0]

print("训练集输入规模", train_X.shape, "训练集标签规模", train_Y.shape)
print("测试集输入规模", test_X.shape, "测试集标签规模", test_Y.shape)

'''
开始生成模拟的电阻噪声-正态分布
'''
miu = 0.0013244880773251254
sigma = 0.0005321991698968055

random_variable = np.random.normal(loc=miu, scale=sigma, size=(len(train_X), 750))
new_trainX = train_X + random_variable
new_trainY = train_Y
random_variable = np.random.normal(loc=miu, scale=sigma, size=(len(test_X), 750))
new_testX = test_X + random_variable
new_testY = test_Y
for i in range(9):
    random_variable = np.random.normal(loc=miu, scale=sigma, size=(len(train_X), 750))
    new_trainX = np.append(new_trainX, train_X + random_variable, axis=0)
    new_trainY = np.append(new_trainY, train_Y)

    random_variable = np.random.normal(loc=miu, scale=sigma, size=(len(test_X), 750))
    new_testX = np.append(new_testX, test_X + random_variable, axis=0)
    new_testY = np.append(new_testY, test_Y)

plt.hist(random_variable[0])
plt.show()
plt.hist(np.mean(random_variable, axis=0), bins=20, edgecolor='grey', )
plt.xlim(0, 0.004)
plt.ylim(0, 130)
plt.show()

print("新训练集输入规模", new_trainX.shape, "新训练集标签规模", new_trainY.shape)
print("新测试集输入规模", new_testX.shape, "新测试集标签规模", new_testY.shape)

scipy.io.savemat(newdataname, {'train_X': new_trainX, 'train_Y': new_trainY, 'test_X': new_testX, 'test_Y': new_testY})
