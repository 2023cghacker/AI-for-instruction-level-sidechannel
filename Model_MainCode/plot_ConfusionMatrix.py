import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import src.Configuration_matplot
import matplotlib.colors as mcolors


def plot_ConfusionMatrix(test_y, predict_y):
    # 获取不同坐标点的数量以及预测结果正确与否
    coord_count = {}
    correct_pred = {}

    for i, (coord, pred) in enumerate(zip(zip(predict_y, test_y), predict_y)):
        if coord in coord_count:
            coord_count[coord] += 1
            if pred == test_y[i]:  # 修改处，使用索引来访问 test_y
                correct_pred[coord] += 1
        else:
            coord_count[coord] = 1
            if pred == test_y[i]:  # 修改处，使用索引来访问 test_y
                correct_pred[coord] = 1

    # 将坐标和数量拆分成两个列表
    coordinates, counts = zip(*coord_count.items())

    # 统计预测正确的和预测错误的
    correct_pred = []
    correct_counts = []
    wrong_pred = []
    wrong_counts = []
    for i, coord in enumerate(coordinates):
        x, y = coord  # 获取坐标的 x 和 y 值
        if x == y:
            correct_pred.append(coord)
            correct_counts.append(counts[i])
        else:
            wrong_pred.append(coord)
            wrong_counts.append(counts[i])

    # print(correct_pred, correct_counts)
    # print(wrong_pred, wrong_counts)

    plt.figure()
    for i, (x, y) in enumerate(correct_pred):
        if i == 0:
            plt.scatter(x, y, color='blue', label='Correct prediction', marker='s')
        plt.scatter(x, y, color='blue', s=correct_counts[i] * 15, marker='s')
        plt.text(x, y, correct_counts[i], ha='center', va='center')

    for i, (x, y) in enumerate(wrong_pred):
        if i == 0:
            plt.scatter(1, 2, color='red', label='wrong prediction', marker='s')
        plt.scatter(x, y, color='red', s=wrong_counts[i] * 10, marker='s')
        plt.text(x, y, wrong_counts[i], ha='center', va='center')

    plt.ylabel('Predicted Label')
    plt.xlabel('True Label')
    plt.xlim(np.min(test_y) - 1, np.max(predict_y) + 1)
    plt.ylim(np.min(test_y) - 1, np.max(predict_y) + 1)
    plt.title('Confusion Matrix')  # 自定义图例和内容
    plt.legend()  # 添加图例，指定标题和位置
    plt.show()


def plot_InstructionMatrix(test_y, predict_y):
    # 获取不同坐标点的数量以及预测结果正确与否
    coord_count = {}
    correct_pred = {}
    M = test_y.shape[0] / 8
    print(M)
    for i, (coord, pred) in enumerate(zip(zip(predict_y, test_y), predict_y)):
        if coord in coord_count:
            coord_count[coord] += 1
            if pred == test_y[i]:  # 修改处，使用索引来访问 test_y
                correct_pred[coord] += 1
        else:
            coord_count[coord] = 1
            if pred == test_y[i]:  # 修改处，使用索引来访问 test_y
                correct_pred[coord] = 1

    # 将坐标和数量拆分成两个列表
    coordinates, counts = zip(*coord_count.items())

    # 统计预测正确的和预测错误的
    correct_pred = []
    correct_counts = []
    wrong_pred = []
    wrong_counts = []
    for i, coord in enumerate(coordinates):
        x, y = coord  # 获取坐标的 x 和 y 值
        if x == y:
            correct_pred.append(coord)
            correct_counts.append(counts[i])
        else:
            wrong_pred.append(coord)
            wrong_counts.append(counts[i])

    # print(correct_pred, correct_counts)
    # print(wrong_pred, wrong_counts)

    plt.figure()

    correct_color = (184 / 255, 168 / 255, 207 / 255)
    wrong_color = (33 / 255, 113 / 255, 181 / 255)

    for i, (x, y) in enumerate(correct_pred):
        color = correct_color  # tuple(M/correct_counts[i] * x for x in correct_color)
        plt.scatter(x, y, color=color, s=1200)
        # plt.text(x, y, correct_counts[i], ha='center', va='center')

    for i, (x, y) in enumerate(wrong_pred):
        # color = tuple(wrong_counts[i] / M * x for x in wrong_color)
        plt.scatter(x, y, color=wrong_color, s=wrong_counts[i])
        # plt.text(x, y, wrong_counts[i], ha='center', va='center')

    font = FontProperties(family='Times New Roman', size=13)
    custom_ticks = [0, 1, 2, 3, 4, 5, 6, 7]  # 自定义的刻度位置
    custom_labels = ['AND', 'ORR', 'EOR', 'BIC', 'CMP', 'CMN', 'TST', 'TEQ']  # 自定义的标签
    plt.xticks(custom_ticks, custom_labels, fontproperties=font)  # 设置x轴的刻度和标签
    plt.yticks(custom_ticks, custom_labels, fontproperties=font)  # 设置y轴的刻度和标签

    font = FontProperties(family='Times New Roman', size=16, weight='bold')
    plt.ylabel('Predicted Label', fontproperties=font)
    plt.xlabel('True Label', fontproperties=font)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(np.min(test_y) - 0.5, np.max(predict_y) + 0.5)
    plt.ylim(np.min(test_y) - 0.5, np.max(predict_y) + 0.5)
    # plt.legend()  # 添加图例，指定标题和位置
    plt.tight_layout()
    plt.show()

# def plot_InstructionMatrix(test_y, pred_y):
#     m = 7
#     data = np.zeros((m, m))
#     # 遍历test_y和pred_y
#     for t, p in zip(test_y, pred_y):
#         # 验证t和p的值是否在0到m-1的范围内
#         if 0 <= t < m and 0 <= p < m:
#             # 增加对应位置的计数
#             data[t, p] += 1
#         else:
#             # 如果t或p超出范围，可以选择忽略或抛出异常
#             # 这里我们选择打印一条消息（实际使用时可能需要根据情况调整）
#             print(f"Warning: Label {t} or {p} is out of range [0, {m - 1}].")
#
#     # data = [[96.375, 48.962, 51.387, 28.181, 64.638, 67.744, 45.075],
#     #         [27.819, 97.938, 47.350, 45.619, 30.475, 30.112, 24.562],
#     #         [27.806, 46.019, 96.562, 39.506, 27.600, 27.881, 24.988],
#     #         [12.637, 25.344, 17.350, 98.688, 16.394, 16.481, 11.756],
#     #         [78.188, 42.181, 40.531, 31.656, 83.469, 73.731, 53.194],
#     #         [80.163, 47.319, 41.337, 45.006, 69.612, 97.344, 53.394],
#     #         [56.519, 33.444, 37.231, 25.475, 55.938, 58.338, 86.938]]
#     # data = np.array(data)
#     print(data.shape)
#     print(data)
#
#     # 创建一个图形和坐标轴
#     fig, ax = plt.subplots()
#     im = ax.imshow(100 - data, cmap='gray', vmin=0, vmax=1, aspect='auto')
#     # 使用imshow绘制数据矩阵，cmap参数选择颜色映射，这里使用'gray'表示黑白灰度
#     # vmin和vmax参数可以指定颜色映射的数值范围，这里使用0和1
#     # aspect='auto'是为了保持格子的正方形形状，但在某些情况下可能需要调整为'equal'
#
#     # 隐藏坐标轴刻度线（但保留刻度标签的显示能力）
#     ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=True,
#                    labelleft=True)
#
#     # 隐藏整个坐标轴框架（可选，如果您不希望看到任何坐标轴线条）
#     # ax.axis('off')  # 如果您想保留刻度标签，请不要取消注释这行代码
#
#     # 设置字体属性
#     font_properties = {'family': 'Times New Roman', 'size': 12, 'weight': 'bold', 'color': 'black'}
#
#     # 添加自定义的轴标签（调整位置以避免遮挡）
#     ax.text(0.5, -0.1, 'Testing Device Label', horizontalalignment='center', transform=ax.transAxes,
#             fontdict=font_properties)
#     ax.text(-0.1, 0.5, 'Training Device Label', verticalalignment='center', rotation='vertical', transform=ax.transAxes,
#             fontdict=font_properties)
#     ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7])
#     ax.set_yticklabels([0, 1, 2, 3, 4, 5, 6, 7])
#     ax.spines['top'].set_color('none')
#     ax.spines['bottom'].set_color('none')
#     ax.spines['left'].set_color('none')
#     ax.spines['right'].set_color('none')
#
#     # 如果你想要在每个格子上显示数值，可以使用下面的代码
#     for i in range(7):
#         for j in range(7):
#             ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='white')
#
#     # 显示图形
#     plt.show()
