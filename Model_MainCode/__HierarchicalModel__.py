"""
    @Author : ling chen
    @Create : 2023/02
    @Last modify: 2023/07
    @Description: 层次模型的训练测试: 可以对一个样本进行分层识别，识别指令类型，再识别寄存器
"""

import time

import joblib
import numpy as np
from src.Configuration_01 import DIRECTORY_PREFIX
from scipy.io import loadmat
from src.Model_MainCode.MYAPI_sklearn.BPNN import CreateBPNN, LoadBPNN
from src.Configuration_03 import DIRECTORY_PREFIX, INSTRUCTION_NAME, INSTRUCTION_REGNAME, INSTRUCTION_REGLABEL, \
    INSTRUCTION_LABEL
from src.Model_MainCode.accuracy import Accuracy


def First_level_classifier(train_x, train_label, test_x, test_label, FIRST_NAME, FIRST_LABEL,
                           method="BPNN"):
    """
    @第一层识别器
    :param INSTRUCTION_NAME: 标签的英文名称
    :param INSTRUCTION_LABEL: 标签的数字编号
    :param method: 选择使用的算法
    """

    # 基本信息输出
    print("\n=================================First_level_classifier=================================")
    print("本层使用的算法是", method)
    print("训练集规模；", np.shape(train_X))
    print("测试集规模；", np.shape(test_X))

    # 开始训练和泛化
    predict_y = LoadBPNN(DIRECTORY_PREFIX + "1st_0726_074042_model.m", test_X)
    # bpnn,predict_y = CreateBPNN(train_x, train_label, test_x, layersize=1000, t=1000)
    # print(predict_y)
    Accuracy(predict_y, test_label, 2, FIRST_NAME, FIRST_LABEL)

    # 保存bp神经网络和训练参数
    # date = time.strftime('%m%d_%H%M%S', time.localtime())  # %Y年份，M月份以此类推
    # model_name = DIRECTORY_PREFIX + date + "_1st_model.m"
    # joblib.dump(bpnn, model_name)

    # 对下一层标签进行分类
    index_matrix = []
    for i in range(len(INSTRUCTION_LABEL)):
        label = INSTRUCTION_LABEL[i]
        index = np.where(predict_y == label)[0]  # 被预测的指令标签所处索引

        print("在本层中，以下样本被分类到指令（", FIRST_NAME[i], "）")
        print(index)  # index的type是ndarry
        index_matrix.append(index.tolist())

    # print(index_matrix)
    return index_matrix


def Second_level_classifier(train_x, train_label, test_x, test_label, First_NAME, Second_NAME,
                            Second_LABEL,
                            method="BPNN"):
    """
    @第二层识别器
    :param INSTRUCTION_NAME: 标签的英文名称
    :param INSTRUCTION_LABEL: 标签的数字编号
    :param method: 选择使用的算法
    """

    # 基本信息输出
    print("\n=================================Second_level_classifier=================================")

    print("正在对上一层中被分类到指令 ", First_NAME, " 的样本进行寄存器识别")
    print("本层使用的算法是", method)
    print("训练集规模；", np.shape(train_x))
    print("测试集规模；", np.shape(test_x))

    # 开始训练和泛化
    bpnn, predict_y = CreateBPNN(train_x, train_label, test_x, layersize=10, t=10)
    # print(predict_newy)
    acc = Accuracy(predict_y, test_label, 2, Second_NAME, Second_LABEL)
    acc_num = acc * len(test_x)  # 预测正确数目

    # 保存bp神经网络和训练参数
    # model_name = DIRECTORY_PREFIX + "2st_" + First_NAME + "_model.m"
    # joblib.dump(bpnn, model_name)

    return acc_num


'''
主函数
'''
# 处理数据集
add_pre = DIRECTORY_PREFIX
matdata = loadmat(add_pre + '100d_train_test_0')  # 读取Mat文件, 文件里有训练集和测试集
train_X = matdata['train_X']
train_Y = matdata['train_Y'][0]
train_Y2 = matdata['train_Y2'][0]
test_X = matdata['test_X']
test_Y = matdata['test_Y'][0]
test_Y2 = matdata['test_Y2'][0]
n1 = np.size(train_X, 0)
n2 = np.size(test_X, 0)
m = np.size(train_X, 1)
print("训练集总规模；", n1, "*", m)
print("测试集总规模；", n2, "*", m)


# 1.第一层识别器
index_matrix = First_level_classifier(train_X, train_Y2, test_X, test_Y2, INSTRUCTION_NAME, INSTRUCTION_LABEL)

# 1.第二层识别器
total_acc = 0
for i in range(len(index_matrix)):
    number = [5600, 6400, 6400, 6400, 6400, 6400, 6400, 6400, 6400, 6400, 6400, 6400, 6400, 6400]  # 第一层每个标签对应的训练集样本数量
    train_newx = train_X[sum(number[0:i]):sum(number[0:i + 1])]
    train_newlabel = train_Y[sum(number[0:i]):sum(number[0:i + 1])]
    index = index_matrix[i]
    test_newx = test_X[index]
    test_newlabel = test_Y[index]

    acc_num = Second_level_classifier(train_newx, train_newlabel, test_newx, test_newlabel, INSTRUCTION_NAME[i],
                                      INSTRUCTION_REGNAME,
                                      INSTRUCTION_REGLABEL)
    total_acc = total_acc + acc_num

print("总准确率为", round(total_acc / n2, 2), "%")
