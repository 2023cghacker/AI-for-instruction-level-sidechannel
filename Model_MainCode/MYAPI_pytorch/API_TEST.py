from scipy.io import loadmat
from src.Configuration_03 import DIRECTORY_PREFIX, INSTRUCTION_REGNAME, INSTRUCTION_REGLABEL, INSTRUCTION_NAME, \
    INSTRUCTION_LABEL
from src.Model_MainCode.MYAPI_pytorch.__CNN_API__ import __CNN_API__, __load_cnnmodel__
from src.Model_MainCode.MYAPI_pytorch.__MLP_API__ import __MLP_API__
import torch
from src.Model_MainCode.accuracy import Accuracy

print(DIRECTORY_PREFIX)
'''
导入数据
'''
DataName = "../../DataFile/DataSets_registerbased/" + "1T_traintest_m(cwt100s).mat"  # 数据集文件名称
matdata = loadmat(DataName)
train_X = matdata['train_X']
train_Y = matdata['train_Y2'][0]
test_X = matdata['test_X']
test_Y = matdata['test_Y2'][0]
print("训练集输入规模", train_X.shape, "训练集标签规模", train_Y.shape)
print("测试集输入规模", test_X.shape, "测试集标签规模", test_Y.shape)

''' 
调用API
'''
dict = {"batch_size": 128, "Epoch": 10, "learning_rate": 1e-4, "output_dim": 15}  # 设置参数
percentage, predict_Y, model = __CNN_API__(train_X, train_Y, test_X, dict)
Accuracy(predict_Y, test_Y, 2, INSTRUCTION_NAME, INSTRUCTION_LABEL)
torch.save(model.state_dict(), "../../DataFile/DataSets_registerbased/1Tm_CNNmodel.pth")  # 保存模型

'''
调用模型
'''
percentage, predict_Y = __load_cnnmodel__(test_X, '../../DataFile/DataSets_registerbased/model.pth')
print(predict_Y)
# Accuracy(predict_Y, test_Y, 2, INSTRUCTION_REGNAME, INSTRUCTION_REGLABEL)


# DataName = DIRECTORY_PREFIX + '1Tl_train_test_0'
# matdata = loadmat(DataName)
# train_X = matdata['train_X']
# train_Y = matdata['train_Y'][0]
# test_X = matdata['test_X']
# test_Y = matdata['test_Y'][0]
# print("训练集输入规模", train_X.shape, "训练集标签规模", train_Y.shape)
# print("测试集输入规模", test_X.shape, "测试集标签规模", test_Y.shape)
#
#
# dict = {"outputdim": 148, "lr": 1e-2, "batch_size": 256}
# percentage, predict_Y = __MLP_API__(train_X, train_Y, test_X, parameters=dict)
# Accuracy(predict_Y, test_Y, 2, INSTRUCTION_REGNAME, INSTRUCTION_REGLABEL)
