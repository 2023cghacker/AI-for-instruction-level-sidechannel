import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 创建数据
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# 100, 400, 900, 1600, 2500, 4900, 10000, 14400, 16900
y1 = [67.194, 74.573, 79.163, 82.312, 85.55, 89.844, 92.894, 89.097, 89.01]
y2 = [63.194, 78.573, 85.546, 86.342, 87.55, 90.844, 93.784, 91.097, 90.17]
y3 = [61.194, 70.573, 81.163, 88.312, 89.55, 91.844, 92.001, 96.097, 94.13]

# 绘制曲线
plt.plot(x, y1, color='#FFC107', label='1st device combination')  # 绘制sin(x)并添加图例
plt.scatter(x, y1, s=30, color='#FFC107', facecolors='none')
plt.scatter(7, y1[6], s=40, color='#FFC107', facecolors='#FFC107')

plt.plot(x, y2, color='#007BFF', label='2nd device combination')  # 绘制cos(x)并添加图例
plt.scatter(x, y2, s=30, color='#007BFF', facecolors='none', )
plt.scatter(7, y2[6], s=40, color='#007BFF', facecolors='#007BFF')

plt.plot(x, y3, color='#DC3545', label='3rd device combination')  # 绘制exp(-x)并添加图例
plt.scatter(x, y3, s=30, color='#DC3545', facecolors='none', )
plt.scatter(8, y3[7], s=40, color='#DC3545', facecolors='#DC3545')
# 添加图例
plt.legend()

# 添加标题和坐标轴标签
font = FontProperties(family='Times New Roman', size=14, weight='bold')
plt.xlabel('K value', fontproperties=font)
plt.ylabel('RA(%)', fontproperties=font)
# plt.ylim([0, 100])

custom_ticks = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 自定义的刻度位置
custom_labels = ['10^2', '20^2', '30^2', '40^2', '50^2', '70^2', '100^2', '120^2', '130^2']  # 自定义的标签
plt.xticks(custom_ticks, custom_labels)  # 设置x轴的刻度和标签
# 隐藏顶部和右侧的边框
ax = plt.gca()  # 获取当前Axes对象
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 显示网格
plt.grid(True, linestyle='--')
plt.tight_layout()

# 显示图形
plt.show()
