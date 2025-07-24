"""
    @Author : ling chen
    @Create : 2023/02
    @Last modify: 2023/04
    @Description: 计算准确率，输入：预测值，真实值，  输出：准确率 以及其他指标。
"""

import numpy as np


def Accuracy(predict_y, test_y, flag, reallabel_list, numlabel_list):
    """
    :param predict_y: 预测的标签值
    :param test_y: 测试集实际的标签值
    :param flag: 标志位，用来选择在控制台输出的内容,1仅输出准确率,2额外输出出现的每个指令准确率,3输出所有信息
    :param reallabel_list: 列表:所有标签的英文名称,一般是 src.Configuration_data_info中的 INSTRUCTION_NAME或 INSTRUCTION_REGNAME
    :param numlabel_list: 列表:所有标签的数字名称,一般是 src.Configuration_data_info中的 INSTRUCTION_LABEL
    :return: 返回预测准确率，并且输出相关数据
    """
    test_y = np.trunc(test_y).astype(int)  # 化为整数
    m = len(reallabel_list)  # reallabel_list是标签对应的指令名称，m是标签的数量
    n = len(test_y)

    # 1.检验预测标签的数量是否正确,出错则结束并返回
    if len(test_y) != len(predict_y):
        print(f"\nERROR!!!  预测的标签数量{len(predict_y)}和测试集的数量{len(test_y)}不匹配!!!\n")
        return

    # 2.检验预测标签的范围是否超出,出错则结束并返回
    if np.min(predict_y) < np.min(numlabel_list) or np.max(predict_y) > np.max(numlabel_list):  # 预测值范围超出
        print("\nWARNING!  预测的标签值超出了标签值实际范围,请检查训练集和测试集是否匹配!!!\n")

    # 3.检验预测标签的范围是否超出,出错则警告
    if np.min(predict_y) < np.min(test_y) or np.max(predict_y) > np.max(test_y):  # 预测值范围超出
        print("\nWARNING! 预测的标签值超出了测试集的标签值范围，请检查是否存在影响\n")
        print([np.min(predict_y), np.max(predict_y)], [np.min(test_y), np.max(test_y)])

    # 4.开始计算准确率
    correct = np.zeros(m)  # m维向量，存储每个指令被预测正确的次数
    predicted = np.zeros(m)  # m维向量，存储每个指令被预测的次数
    total = np.zeros(m)  # m维向量，存储每个指令实际出现的次数
    accuracy = 0  # 总准确率
    for i in range(n):
        if predict_y[i] == test_y[i]:
            accuracy = accuracy + 1  # 预测正确加一

        if flag != 1:
            index1 = np.where(numlabel_list == test_y[i])[0][0]  # 实际的指令标签所处索引
            index2 = np.where(numlabel_list == predict_y[i])[0][0]  # 被预测的指令标签所处索引
            total[index1] = total[index1] + 1  # 该指令实际出现
            predicted[index2] = predicted[index2] + 1  # 该指令被预测到
            if index1 == index2:
                correct[index1] = correct[index1] + 1  # 该指令被预测正确

    accuracy = accuracy / n * 100
    print("\n======== 总预测准确率为", round(accuracy, 3), '% ========')

    if flag == 2:  # 如果要求仅输出出现的指令标签各自的准确指标
        print('%+30s\t%+10s\t%+10s\t' % ("实际总数", "召回率（查全）", "精确率（查准）"))
        for i in range(m):
            if total[i] != 0 and predicted[i] != 0:
                print('%+15s\t' % reallabel_list[i], end='')
                print('%+15s\t' % int(total[i]), end='')
                print('%+15s\t' % str(round(correct[i] / total[i] * 100, 2)) + "%", end='')
                print('%+15s\t' % str(round(correct[i] / predicted[i] * 100, 2)) + "%")

    if flag == 3:  # 如果要求输出全部指令标签各自的准确指标
        print('%+30s\t%+10s\t%+10s\t' % ("实际总数", "召回率（查全）", "精确率（查准）"))
        for i in range(m):
            if total[i] != 0 and predicted[i] != 0:
                print('%+15s\t' % reallabel_list[i], end='')
                print('%+15s\t' % int(total[i]), end='')
                print('%+15s\t' % str(round(correct[i] / total[i] * 100, 2)) + "%", end='')
                print('%+15s\t' % str(round(correct[i] / predicted[i] * 100, 2)) + "%")
            else:
                print('%+15s\t' % reallabel_list[i], end='')
                print('%+15s\t' % "NONE", end='')
                print('%+15s\t' % "NONE", end='')
                print('%+15s\t' % "NONE")

    print("\n")
    return accuracy

# Accuracy([1], [1])
