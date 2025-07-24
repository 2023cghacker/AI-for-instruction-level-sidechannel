import time
import numpy as np
from scipy.io import loadmat

from src.Model_MainCode.Recognition_Rate import Recognition_Rate
from src.Model_MainCode.MYAPI_sklearn.BPNN import CreateBPNN
from src.Configuration_01 import DIRECTORY_PREFIX

begint = time.time()

# ============1、导入训练集和测试数据================
address = DIRECTORY_PREFIX + 'train_test_0'
matdata = loadmat(address)  # 读取Mat文件, 文件里有训练集和测试集
train_X = matdata['train_X']
train_Y = matdata['train_Y'][0]
n1 = np.size(train_X, 0)
m = np.size(train_X, 1)  # m=125
print("训练集规模；", n1, "*", m)

address = DIRECTORY_PREFIX + 'long_segment_01'
matdata = loadmat(address)  # 读取Mat文件
Power_segment = matdata['Power_segment'][0]
Instruction_segment = matdata['Instruction_segment']
Instruction_id = matdata['Instruction_id'][0]
Instruction_index = matdata['Instruction_index'][0]

print("功率片段长度为：", len(Power_segment), ",其中包含了", len(Instruction_id), "指令")
# print(Power_segment,Instruction_segment,Instruction_id)

test_X = np.zeros((len(Power_segment) - 125, 125))
for i in range(len(Power_segment) - 125):
    Powerx = Power_segment[i:i + 125]
    test_X[i] = Powerx  # 拼接
# print(test_X)
print("测试集规模：", test_X.shape)

# ============2、选用模型进行训练和测试============
predict_Y = CreateBPNN(train_X, train_Y, test_X, layersize=500, t=500)

# 比较指令识别率，计算准确率等
Recognition_Rate(predict_Y, Instruction_index, Instruction_id, Instruction_segment)

# print("\n预测出了", count, "个指令")
# print("实际指令流：")
# for i in range(len(Instruction_segment)):
#     print(int(Instruction_index[i]), Instruction_segment[i], end="  ")


# Accuracy(predict_Y, test_Y, "all")

#
# endt = time.time()
# print("本次运行所花时间为：", endt - begint, "s")
