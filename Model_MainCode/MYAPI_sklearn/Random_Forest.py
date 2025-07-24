"""
随机森林
"""
import numpy as np
from scipy.io import loadmat
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

from src.Model_MainCode.accuracy import Accuracy
from src.Configuration_01 import DIRECTORY_PREFIX

# 读取数据
address = DIRECTORY_PREFIX + '20d_(L)train_test_0'
# address = "train_test_0"
matdata = loadmat(address)  # 读取Mat文件, 文件里有训练集和测试集
train_X = matdata['train_X']
train_Y = matdata['train_Y'][0]
test_X = matdata['test_X']
test_Y = matdata['test_Y'][0]
n1 = np.size(train_X, 0)
n2 = np.size(test_X, 0)
m = np.size(train_X, 1)  # m=125
print("训练集规模；", n1, "*", m)
print("测试集规模；", n2, "*", m)

# 模型训练和测试
'''
参数n_estimators=300，设置了树的个数
'''
rf = RandomForestRegressor(n_estimators=300, random_state=42)
rf.fit(train_X, train_Y)
y_pred = rf.predict(test_X)
Accuracy(y_pred , test_Y, "all")
print(y_pred)
mae = mean_absolute_error(test_Y, y_pred)
print("MAE:", mae)
# 结果绘图
plt.plot(test_Y, label='Test_Y', )
plt.plot(y_pred, color='red', label='Predictions')
plt.legend(loc='upper left')
plt.show()
