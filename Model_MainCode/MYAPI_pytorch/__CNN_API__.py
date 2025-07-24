"""
    @Author : ling chen
    @Create : 2023/11
    @Last modify: 2023/11
    @Description:PYTORCH-cnn接口
"""
import time

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from src.Configuration_03 import DIRECTORY_PREFIX
from src.Model_MainCode.accuracy import Accuracy
import src.Configuration_matplot


class MytrainDataset(Dataset):
    """
    继承父类 torch.utils.data.Dataset
    """

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        pass

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]

    def __len__(self):
        return len(self.y_train)


class MytestDataset(Dataset):
    """
    继承父类 torch.utils.data.Dataset
    """

    def __init__(self, x_train):
        self.x_train = x_train
        pass

    def __getitem__(self, idx):
        return self.x_train[idx]

    def __len__(self):
        return len(self.x_train)


# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self, H, M, output_dim):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),  # 卷积层
            nn.ReLU(),  # 激励层
            nn.MaxPool2d(kernel_size=2, stride=2)  # 池化层
        )

        self.layer1 = nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer2 = nn.BatchNorm2d(num_features=32)

        # 初始化 Linear 层，输入维度暂时设置为 None
        self.out = nn.Linear(32 * int(H / 4) * int(M / 4), output_dim)  # 全连接层28800

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.conv2(x)
        x = self.layer2(x)

        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


def __CNN_API__(train_X, train_Y, test_X, parameters):
    """
    :param train_X: 训练集，二维的样本
    :param train_Y: 训练集标签
    :param test_X: 测试集
    :param parameters: 参数，包括batch_size、outputdim、lr、epoch
    :return:
    """
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    1、对数据集进行归一化和格式转换(数组->tensor)
      然后加载数据集
    '''
    H, M = train_X.shape[1], train_X.shape[2]
    print(H, M)
    train_X = torch.Tensor(train_X).unsqueeze(1).to(device)  # 将输入的维度从 (n, a, b) 调整为 (n, 1, a, b)
    train_Y = torch.LongTensor(train_Y).to(device)
    train_dataset = MytrainDataset(train_X, train_Y)  # 使用实际的训练集输入和标签
    train_loader = DataLoader(dataset=train_dataset, batch_size=parameters["batch_size"], shuffle=True, drop_last=True)

    test_X = torch.Tensor(test_X).unsqueeze(1).to(device)
    test_dataset = MytestDataset(test_X)  # 使用实际的训练集输入和标签
    test_loader = DataLoader(dataset=test_dataset, batch_size=parameters["batch_size"], shuffle=False, drop_last=False)

    '''
    2.加载模型和优化器
    '''
    # 初始化模型和优化器
    model = CNN(H, M, parameters["outputdim"]).to(device)  # 模型移到GPU
    print("\n", model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=parameters["lr"])  # 优化器

    '''
    3.开始在模型上进行训练
    '''
    total_step = len(train_loader)
    loss_list = []

    print("\n==========================training==============================\n")
    for epoch in range(parameters["epoch"]):
        for i, (x, y) in enumerate(train_loader):
            predict = model(x)
            loss = criterion(predict, y)  # 将标签调整为与输出维度匹配

            '''
            三件套
            '''
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 清理无用的中间变量
            del x, y, predict
            torch.cuda.empty_cache()

            if (i + 1) % 5 == 0:
                print(f'epoch [{epoch + 1}/{parameters["epoch"]}], Step [{i + 1}/{total_step}], Loss: {loss.item()}')

        with torch.no_grad():
            loss_list.append(loss.tolist())
            plt.cla()
            plt.plot(loss_list, 'r-', lw=1)  # 直接输入y轴坐标，不输入x轴坐标是可以的
            plt.ylabel('Loss')
            plt.title('loss=%.4f step=%d' % (loss.cpu(), epoch))
            plt.pause(0.1)

    print("\n=========================training ended==========================\n")

    '''
    使用训练好的模型进行泛化测试
    '''
    all_predictions = []
    with torch.no_grad():
        for x in test_loader:  # 假设我们不需要标签进行预测
            predict = model(x)
            all_predictions.append(predict)

    # 将所有预测结果拼接起来
    y_predict = torch.cat(all_predictions, dim=0)
    print(f"预测值:{y_predict}")

    # 将输出通过softmax变为概率值
    percentage = torch.softmax(y_predict, dim=1)
    percentage = percentage.cpu().detach().numpy()  # Tensor转数组
    percentage = np.round(percentage * 100, 3)  # 乘以100为百分比，保留两位小数
    np.set_printoptions(suppress=True)  # 关闭Numpy的科学计数法
    print("预测标签概率:")
    print(percentage, "%")

    # 选择概率值最大的下标作为预测标签
    result = np.argmax(percentage, 1)
    print("预测标签:")
    print(result)

    if parameters["saveflag"]:
        # 保存网络和训练参数
        date = time.strftime('%m%d_%H%M', time.localtime())  # %Y年份，M月份以此类推
        model_name = date + "_cnn.pth"
        torch.save(model.state_dict(), model_name)  # 保存模型
        print("神经网络保存在", model_name, "文件中")

    # 释放内存
    del model
    torch.cuda.empty_cache()

    return percentage, result


def __load_cnnmodel__(test_X, modeladdress):
    H = 200
    M = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN(H, M, 16)  # 模型移到GPU
    model.load_state_dict(torch.load(modeladdress))
    print(model)

    test_X = torch.Tensor(test_X)
    test_X = test_X.unsqueeze(1)
    print(test_X.shape, type(test_X))
    y_predict = model(test_X)

    # 将输出通过softmax变为概率值
    percentage = torch.softmax(y_predict, dim=1)
    percentage = percentage.detach().numpy()  # Tensor转数组
    percentage = np.round(percentage * 100, 3)  # 乘以100为百分比，保留两位小数
    print("预测标签概率:")
    print(percentage, "%")
    # 选择概率值最大的下标作为预测标签
    result = np.argmax(percentage, 1)
    print("预测标签:")
    print(result)

    return percentage, result
