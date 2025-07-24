from matplotlib import pyplot as plt

from src.Configuration_05 import DIRECTORY_PREFIX
from src.Model_MainCode.Loadmatfile import loadCWT_stm32, loaddevice

X1, train_Y_i, train_Y_d = loadCWT_stm32([1], DIRECTORY_PREFIX + 'DATA_(cwt500s)_{')
X2, train_Y_i, train_Y_d = loaddevice([1, 2, 3, 4, 5, 6], 500, DIRECTORY_PREFIX + 'DATA_(200d500s).mat')
# 创建图形和轴
# plt.figure(figsize=(10, 5))
font_properties = {'family': 'Times New Roman', 'size': 12, 'weight': 'bold', 'color': 'black'}

plt.imshow(X1[0], cmap='viridis', aspect='auto')  # 使用 imshow 显示二维数组
# 添加颜色条
plt.ylabel('Time', fontsize=18, fontweight='bold', family='Times New Roman', color='black')
plt.xlabel('Frequency', fontsize=18, fontweight='bold', family='Times New Roman', color='black')
plt.colorbar()
plt.tight_layout()
plt.show()

plt.plot(X2[2])
plt.xlabel('Time', fontsize=18, fontweight='bold', family='Times New Roman', color='black')
plt.tight_layout()
plt.show()
