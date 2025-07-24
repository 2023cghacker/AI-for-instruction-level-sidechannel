import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

# 生成一个随机矩阵，例如10x10
matrix = [[322, 1, 0, 0, 78, 0, 0, ],
          [2, 276, 6, 3, 4, 115, 7, ],
          [2, 5, 251, 9, 2, 1, 125, ],
          [0, 4, 3, 251, 2, 2, 4, ],
          [73, 0, 1, 0, 310, 2, 1, ],
          [0, 107, 3, 1, 0, 269, 7, ],
          [0, 7, 129, 5, 1, 9, 249, ]]
matrix = np.array(matrix)

"""
定义自定义颜色映射
"""
cdict = {'red': ((0.0, 32 / 255, 32 / 255),
                 (1.0, 225 / 255, 225 / 255)),

         'green': ((0.0, 56 / 255, 56 / 255),
                   (1.0, 156 / 255, 156 / 255)),

         'blue': ((0.0, 136 / 255, 136 / 255),
                  (1.0, 102 / 255, 102 / 255))}

# 十六进制颜色代码
start_hex = "#D5E1E3" # "#F6C4A8"
end_hex =  "#D93F49"# "#3D82BF"
start_rgb = mcolors.hex2color(start_hex)
end_rgb = mcolors.hex2color(end_hex)

cdict = {
    'red': ((0.0, start_rgb[0], start_rgb[0]),
            (1.0, end_rgb[0], end_rgb[0])),

    'green': ((0.0, start_rgb[1], start_rgb[1]),
              (1.0, end_rgb[1], end_rgb[1])),

    'blue': ((0.0, start_rgb[2], start_rgb[2]),
             (1.0, end_rgb[2], end_rgb[2]))
}

# 创建颜色映射
my_cmap = LinearSegmentedColormap('my_colormap', cdict)

# 绘制矩阵
fig, ax = plt.subplots()
cax = ax.imshow(matrix, cmap=my_cmap, interpolation='nearest', aspect='auto')

# 添加颜色条
plt.colorbar(cax)

# 添加网格线
ax.set_xticks(np.arange(0.5, matrix.shape[1], 1), minor=True)
ax.set_yticks(np.arange(0.5, matrix.shape[0], 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

# 设置x和y轴的标签
ax.set_xticks(np.arange(matrix.shape[1]))
ax.set_yticks(np.arange(matrix.shape[0]))
ax.set_xticklabels(np.arange(1, matrix.shape[1] + 1))
ax.set_yticklabels(np.arange(1, matrix.shape[0] + 1))

# 隐藏坐标轴
ax.axis('on')

# 显示图像
plt.show()
