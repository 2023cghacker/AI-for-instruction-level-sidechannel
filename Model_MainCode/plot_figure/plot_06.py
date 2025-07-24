import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

from src.Configuration_05 import DIRECTORY_PREFIX, INSTRUCTION_NAME, INSTRUCTION_LABEL
from src.Model_MainCode.Loadmatfile import loaddevice
from src.Model_MainCode.MYAPI_pytorch.__BPNN_API__ import __BPNN_API__
from src.Model_MainCode.accuracy import Accuracy
from src.Model_MainCode.plot_ConfusionMatrix import plot_ConfusionMatrix

# train_X, train_Y_i, train_Y_d = loaddevice([2, 3, 4, 5, 6], 1000, DIRECTORY_PREFIX + 'DATA_(200d1000s).mat')
# test_X, test_Y_i, test_Y_d = loaddevice([1], 1000, DIRECTORY_PREFIX + 'DATA_(200d1000s).mat')
#
# # 生成1000个不重复的随机索引
# indices = np.random.randint(0, len(train_X), size=100000)
# train_X = train_X[indices]
# train_Y_d = train_Y_d[indices]
# train_Y_i = train_Y_i[indices]
# dict = {"outputdim": 16, "lr": 1e-4, "batch_size": 64, "epoch": 40, "saveflag": False}
#
# percentage, predict_Y_i = __BPNN_API__(train_X, train_Y_i, test_X, parameters=dict)
# Accuracy(predict_Y_i, test_Y_i, 2, INSTRUCTION_NAME, INSTRUCTION_LABEL)  # 准确率
# plot_ConfusionMatrix(predict_Y_i, test_Y_i)  # 分布图

# # 创建数据
x = [5, 7, 10, 20, 30, 40, 50, 60, 70, 100]  # K
y = [5000, 2500, 1500, 1000, 600, 300, 100, 70, 60, 40]

#
# # 绘制曲线
plt.plot(x, y,color='#B83945',label='1st device combination',linewidth=3)  # 绘制sin(x)并添加图例
# plt.scatter(x, y, s=30, color='#B83945', facecolors='none')
plt.fill_between(x, y, 0, color='#F5DFDB', label='Filled area')
# # 添加图例
# plt.legend()
#
# # 添加标题和坐标轴标签
font = FontProperties(family='Times New Roman', size=16, weight='bold')
plt.xlabel('Number of Training Traces', fontproperties=font)
plt.ylabel('Training Epochs', fontproperties=font)
# plt.ylim([0, 100])

custom_ticks = [5, 10, 20, 30, 40, 50, 60, 70, 100]  # 自定义的刻度位置
custom_labels = [ '5K', '10K', '20K', '30K', '40K', '50K', '60K', '70K', '100K']  # 自定义的标签
plt.xticks(custom_ticks, custom_labels,fontsize=14)  # 设置x轴的刻度和标签
custom_ticks = [0, 1000, 2000, 5000]  # 自定义的刻度位置
custom_labels = ['0', '1k', '>2k', '+∞']  # 自定义的标签
plt.yticks(custom_ticks, custom_labels,fontsize=14)  # 设置x轴的刻度和标签

plt.tight_layout()
# 隐藏顶部和右侧的边框
ax = plt.gca()  # 获取当前Axes对象
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 显示网格
plt.grid(True, linestyle='--')

# 显示图形
plt.show()
