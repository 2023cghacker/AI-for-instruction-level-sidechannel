import itertools

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties

from src.Configuration_05_2 import INSTRUCTION_NAME, DIRECTORY_PREFIX, INSTRUCTION_LABEL
from src.Model_MainCode.Loadmatfile import loaddevice
from src.Model_MainCode.MYAPI_pytorch.__BPNN_API__ import __BPNN_API__
from src.Model_MainCode.__DivideData__ import DivideData
from src.Model_MainCode.accuracy import Accuracy
from itertools import permutations


def sum_coordinates(coords):
    # 定义混淆矩阵
    matrix = [
        [96.375, 48.962, 51.387, 28.181, 64.638, 67.744, 45.075],
        [27.819, 97.938, 47.350, 45.619, 30.475, 30.112, 24.562],
        [27.806, 46.019, 96.562, 39.506, 27.600, 27.881, 24.988],
        [12.637, 25.344, 17.350, 98.688, 16.394, 16.481, 11.756],
        [78.188, 42.181, 40.531, 31.656, 83.469, 73.731, 53.194],
        [80.163, 47.319, 41.337, 45.006, 69.612, 97.344, 53.394],
        [56.519, 33.444, 37.231, 25.475, 55.938, 58.338, 86.938]
    ]
    # 减去1以符合Python的0索引
    coords = [(x - 1, y - 1) for x, y in coords]
    total_sum = 0

    # 遍历所有坐标对，并计算对应位置的值的总和
    for (i, j) in coords:
        if i < len(matrix) and j < len(matrix):
            total_sum += matrix[i][j]

    return total_sum


def generate_coordinate_pairs(numbers):
    # 生成所有可能的坐标对
    pairs = set()
    for (i, j) in permutations(numbers, 2):
        pairs.add((i, j))
    return pairs


"""
多个设备
"""
# 定义范围
# numbers = list(range(2, 8))
# # 生成所有组合
# combinations = itertools.combinations(numbers, 4)
#
# # 将每个组合转换为列表并输出
# for combo in combinations:
#     print(list(combo))
#     train_X, train_Y_i, train_Y_d = loaddevice(combo, 500, DIRECTORY_PREFIX + 'DATA_(200d1000s8c).mat')
#     test_X, test_Y_i, test_Y_d = loaddevice([1], 500, DIRECTORY_PREFIX + 'DATA_(200d1000s8c).mat')
#     dict = {"outputdim": 8, "lr": 5e-5, "batch_size": 64, "epoch": 200, "saveflag": False}
#     percentage, predict_Y_i = __BPNN_API__(train_X, train_Y_i, test_X, parameters=dict)
#     Accuracy(predict_Y_i, test_Y_i, 2, INSTRUCTION_NAME, INSTRUCTION_LABEL)  # 准确率
#
#     print("\n\n\n")


"""
单个设备
"""
# for idx in [2, 3, 4, 5, 6, 7]:
#     print(idx)
#     X, Y_i, Y_d = loaddevice([idx], 500, DIRECTORY_PREFIX + 'DATA_(200d1000s8c).mat')
#     train_X, train_Y_i, train_Y_d, test_X, test_Y_i, test_Y_d = DivideData(X=X, Y_i=Y_i, Y_d=Y_d)
#
#     dict = {"outputdim": 8, "lr": 5e-5, "batch_size": 64, "epoch": 200, "saveflag": False}
#     percentage, predict_Y_i = __BPNN_API__(train_X, train_Y_i, test_X, parameters=dict)
#     Accuracy(predict_Y_i, test_Y_i, 2, INSTRUCTION_NAME, INSTRUCTION_LABEL)  # 准确率
#
#     print("\n\n\n")

"""
计算组合准确率
"""
# # 生成所有可能的坐标对
# coordinates = generate_coordinate_pairs([2, 3, 5, 6])
# print(coordinates)
#
# # 计算坐标处的值的总和
# result = sum_coordinates(coordinates)
# print("总和:", result)
# 数据


"""
绘图
"""
# 示例数据
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
acc = [93.537, 92.875, 91.062, 89.9, 87.75, 87.287, 87.237, 85.088, 84.588, 84.475,
       83.225, 80.737, 77.587, 74.312, 66.25]
y = [524.148, 572.300, 563.775, 510.975, 473.930, 471.975, 447.085, 429.324, 463.513,
     410.025, 396.037, 416.85, 381.619, 398.743, 378.644]
y = [a / 12 for a in y]


# 创建图形和坐标轴
fig, ax1 = plt.subplots()

# 设置字体属性
font = FontProperties(family='Times New Roman', size=12)
font_large = FontProperties(family='Times New Roman', size=16, weight='bold')

# 绘制第一条曲线 (acc)
# 定义起始和终止颜色的十六进制值
start_color = '#82A7D1'  # '#E1EEF6' , "#C0A3C0"
end_color = '#82A7D1'

# 创建自定义的颜色映射
cmap = LinearSegmentedColormap.from_list("custom_cmap", [start_color, end_color])
normalize = plt.Normalize(min(acc), max(acc))  # 归一化函数，将数据标准化到 [0, 1] 的范围
colors = cmap(normalize(acc))  # 生成与 acc 对应的颜色

# 自定义刻度和标签
custom_ticks = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]  # 自定义的刻度位置
custom_labels = ['{2,3,5,6}', '{2,5,6,7}', '{3,5,6,7}', '{4,5,6,7', '{2,4,5,6}', '{2,3,6,7}',
                 '{3,4,5,6}', '{2,3,4,6}', '{2,3,5,7}', '{2,3,4,5}', '{2,4,5,7}', '{2,4,6,7}',
                 '{3,4,5,7}', '{3,4,6,7}', '{2,3,4,7}']  # 自定义的标签

# 设置x轴的刻度位置和标签
ax1.set_xticks(custom_ticks)
ax1.set_xticklabels(custom_labels, rotation=45, ha='right', fontproperties=font)  # 斜着显示并右对齐
ax1.tick_params(axis='x', which='both', length=0)  # 去掉刻度线

# 绘制柱状图
ax1.bar(x, acc, color=colors, label='RA')  # 添加图例标签
ax1.set_xlabel('Training Device Set', fontproperties=font_large)
ax1.set_ylabel('RA(%)', fontproperties=font_large)
ax1.set_ylim(60, 100)
ax1.tick_params(axis='y')
ax1.spines['top'].set_visible(False)  # 去掉上方的边框线

# 创建第二个Y轴
ax2 = ax1.twinx()  # 共享x轴

# 绘制第二条曲线 (y)
color = '#BA3E45'  # '#F57C6E', '#5E67AA'
ax2.plot(y, color=color, linewidth=3.5, label='Generalized similarity')  # 添加图例标签
# plt.scatter(x, y, s=30, color=color, facecolors='white')
ax2.plot(x, y, 'o', markersize=10, markerfacecolor='white', markeredgecolor=color, markeredgewidth=2)

ax2.set_ylabel('Generalized similarity', fontproperties=font_large)
ax2.set_ylim(25, 60)
ax2.tick_params(axis='y')
ax2.spines['top'].set_visible(False)

font = FontProperties(family='Times New Roman', size=12)
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.95), prop=font)
# 显示图形
fig.tight_layout()  # 自动调整子图参数，以给图形留出足够的空间
plt.show()
