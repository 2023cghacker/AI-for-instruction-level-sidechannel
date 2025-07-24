import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# 创建数据
x = [1, 2, 3, 4, 5, 6]
y1 = [63.8, 71.7, 75.7, 80.9, 81.3, 83.0]
y2 = [64.0, 69.7, 76.7, 82.0, 84.3, 84.6]
y3 = [61.3, 72.9, 80.2, 83.9, 89.7, 92.8]  # 1,2,3,4,5,6 our work

# 绘制曲线
plt.plot(x, y1, color='#FFC107', label='MLP')  # 绘制sin(x)并添加图例
plt.scatter(x, y1, s=30, color='#FFC107', facecolors='none')
plt.plot(x, y2, color='#007BFF', label='CNN')  # 绘制cos(x)并添加图例
plt.scatter(x, y2, s=30, color='#007BFF', facecolors='none', )
plt.plot(x, y3, color='#DC3545', label='our work')  # 绘制exp(-x)并添加图例
plt.scatter(x, y3, s=30, color='#DC3545', facecolors='none', )
# 添加图例
plt.legend()

# 添加标题和坐标轴标签

font = FontProperties(family='Times New Roman', size=14, weight='bold')
plt.xlabel('Number of Training Devices', fontproperties=font)
plt.ylabel('RA(%)', fontproperties=font)
plt.xlim([0.5, 6.5])
# plt.ylim([40, 100])
plt.box(on=True)
# custom_ticks = [0, 1, 2, 3, 4]  # 自定义的刻度位置
# custom_labels = ['2', '3', '4', '5', '6']  # 自定义的标签
# plt.xticks(custom_ticks, custom_labels)  # 设置x轴的刻度和标签

ax = plt.gca()  # 获取当前Axes对象
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 显示网格
plt.grid(True, linestyle='--')

# 显示图形
plt.show()
