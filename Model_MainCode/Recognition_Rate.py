"""
计算功率片段中指令被识别出来的成功率
author:ling chen
date:2023/7
"""
from src.Configuration_01 import INSTRUCTION_NAME


def Recognition_Rate(predict_Y, Instruction_index, Instruction_id, Instruction_segment):
    count = 0  # 预测出的指令总数
    t = 0
    correct = 0  # 预测正确的指令总数
    print("\n\n预测出的指令流：", end="")
    for i in range(len(predict_Y)):
        if predict_Y[i] != -1:
            print(i, INSTRUCTION_NAME[predict_Y[i]], end="  ")
            count = count + 1
            if (i == Instruction_index[t] and predict_Y[i] == Instruction_id[t]):
                correct = correct + 1

            if i >= Instruction_index[t] and t < len(Instruction_id) - 1:
                t = t + 1
                # print(t)
    print("实际指令流：")
    for i in range(len(Instruction_segment)):
        print(int(Instruction_index[i]), Instruction_segment[i], end="  ")

    print("\n预测出了", count, "个指令")
    print("有", correct, "个指令被正确识别")
    print("识别准确率为：", correct / len(Instruction_id) * 100, "%")
