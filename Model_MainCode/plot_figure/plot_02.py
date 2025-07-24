from scipy.io import loadmat
import matplotlib.colors as mcolors
from src.Configuration_04 import DIRECTORY_PREFIX
from src.Model_MainCode.Loadmatfile import loadData
import numpy as np
import matplotlib.pyplot as plt

X, Y = loadData(DIRECTORY_PREFIX + 'DATA_(CWT750_100).mat')

matdata = loadmat("../MATLAB_CODE/stm32/feature_900.mat")  # 读取Mat文件
X = matdata['B']
# X = np.transpose(X)
indices = matdata['top_K_indices']
print(X.shape, indices.shape)

# 将这些点在图上标注为红色点
y_coords = indices % 100
x_coords = indices // 100

# 绘制平面图
plt.figure(figsize=(6, 6))
plt.imshow(X, cmap='viridis', aspect='auto')
# plt.gca().set_aspect('equal', adjustable='box')
# contour = plt.contour(X, cmap='viridis')
font_properties = {'family': 'Times New Roman', 'size': 16, 'weight': 'bold', 'color': 'black'}
# plt.xlabel('frequency', fontdict=font_properties)
# plt.ylabel('time', fontdict=font_properties)
plt.grid()  # 关闭网格线

plt.scatter(x_coords, y_coords, color='red', label='Selected Points', s=10,
            marker='s', facecolors='none', edgecolors='red')
plt.tight_layout()
plt.show()

# 提取这100个点的数据，并按顺序重新排列为10x10数组
selected_points = X.flatten()[indices].reshape(30, 30)
# 绘制这个10x10的数组的平面图
plt.figure()
# 定义起始颜色和终止颜色的十六进制值
# start_color = '#D03542'  # 蓝色
# end_color = '#314ABC'    # 红色
# # 创建自定义的颜色映射
# cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', [start_color, end_color])

plt.imshow(selected_points, cmap='viridis', aspect='equal')  # 使用viridis色图显示数据
# plt.colorbar()  # 添加颜色条
# plt.title('Selected Points (Reshaped to 10x10)')
plt.grid(False)  # 关闭网格线
plt.tight_layout()
plt.show()
