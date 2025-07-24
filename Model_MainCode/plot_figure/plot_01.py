import numpy as np
import matplotlib.pyplot as plt
from src.Configuration_05 import DIRECTORY_PREFIX, INSTRUCTION_NAME, INSTRUCTION_LABEL
from src.Model_MainCode.Loadmatfile import loadData_stm32
from src.Model_MainCode.MYAPI_pytorch.__BPNN_API__ import __BPNN_API__, __load_bpnnmodel__
from src.Model_MainCode.__DivideData__ import DivideData
from src.Model_MainCode.accuracy import Accuracy

# X, Y_i, Y_d = loadData_stm32(DIRECTORY_PREFIX + 'DATA_(200d1000s).mat')
#
# idx = 7
# train_X, train_Y_i, train_Y_d, test_X, test_Y_i, test_Y_d = DivideData(X=X[(idx - 1) * 16000:idx * 16000, :],
#                                                                        Y_i=Y_i[(idx - 1) * 16000:idx * 16000],
#                                                                        Y_d=Y_d[(idx - 1) * 16000:idx * 16000])
#
# dict = {"outputdim": 16, "lr": 5e-4, "batch_size": 64, "epoch": 200, "saveflag": True}
# percentage, predict_Y_i = __BPNN_API__(train_X, train_Y_i, test_X, parameters=dict)
# Accuracy(predict_Y_i, test_Y_i, 0, INSTRUCTION_NAME, INSTRUCTION_LABEL)  # 准确率
#
# for i in [1, 2, 3, 4, 5, 6]:
#     print(f"device={i}")
#     test_X = X[16000 * (i - 1):16000 * i, :]
#     test_Y_i = Y_i[16000 * (i - 1):16000 * i]
#     percentage, predict_Y = __load_bpnnmodel__(test_X, outputdim=16, modeladdress='device1_bpnn.pth')
#     acc = Accuracy(predict_Y, test_Y_i, 0, INSTRUCTION_NAME, INSTRUCTION_LABEL)  # 准确率
#     print(acc)

"""
绘图
"""
data = [[98.375, 48.962, 51.387, 28.181, 64.638, 67.744, 45.075],
        [27.819, 99.938, 47.350, 45.619, 30.475, 30.112, 24.562],
        [27.806, 46.019, 99.562, 39.506, 27.600, 27.881, 24.988],
        [12.637, 25.344, 17.350, 99.588, 16.394, 16.481, 11.756],
        [78.188, 42.181, 40.531, 31.656, 95.469, 73.731, 53.194],
        [80.163, 47.319, 41.337, 45.006, 69.612, 99.344, 53.394],
        [56.519, 33.444, 37.231, 25.475, 55.938, 58.338, 88.938]]
data = np.array(data)
print(data.shape, np.max(data))

# 创建一个图形和坐标轴
fig, ax = plt.subplots()
im = ax.imshow(100 - data, cmap='gray', vmin=0, vmax=100, aspect='auto')
# 使用imshow绘制数据矩阵，cmap参数选择颜色映射，这里使用'gray'表示黑白灰度
# vmin和vmax参数可以指定颜色映射的数值范围，这里使用0和1
# aspect='auto'是为了保持格子的正方形形状，但在某些情况下可能需要调整为'equal'

# 隐藏坐标轴刻度线（但保留刻度标签的显示能力）
ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=True,
               labelleft=True)

# 隐藏整个坐标轴框架（可选，如果您不希望看到任何坐标轴线条）
# ax.axis('off')  # 如果您想保留刻度标签，请不要取消注释这行代码

# 设置字体属性
font_properties = {'family': 'Times New Roman', 'size': 12, 'weight': 'bold', 'color': 'black'}

# 添加自定义的轴标签（调整位置以避免遮挡）
ax.text(0.5, -0.1, 'Testing Device Label', horizontalalignment='center', transform=ax.transAxes,
        fontdict=font_properties)
ax.text(-0.1, 0.5, 'Training Device Label', verticalalignment='center', rotation='vertical', transform=ax.transAxes,
        fontdict=font_properties)
ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7])
ax.set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7])
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')

# 如果你想要在每个格子上显示数值，可以使用下面的代码
for i in range(7):
    for j in range(7):
        ax.text(j, i, f'{data[i, j]:.2f}%', ha='center', va='center', color='white')

# 显示图形
plt.show()
