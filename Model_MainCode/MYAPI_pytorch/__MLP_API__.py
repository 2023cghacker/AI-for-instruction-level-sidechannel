"""
    @Author : ling chen
    @Create : 2023/07
    @Last modify: 2023/11
    @Description:
"""
import time

import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import src.Configuration_matplot


class MyDataset(Data.Dataset):
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


class MyModel(nn.Module):
    """
    继承父类 torch.nn.Module
    """

    def __init__(self, input_dim, output_dim, dim1, dim2):
        super(MyModel, self).__init__()
        # 一个序列容器，用于搭建神经网络的模块被按照被传入构造器的顺序添加到nn.Sequential()容器
        self.layer = nn.Sequential(

            # nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, dim1),  # 线性层(输入维度，输出维度)
            nn.ReLU(),
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Linear(dim2, output_dim),
            # nn.ReLU()
        )

    def forward(self, x):
        """
            #前向传播
            :param x: 输入矩阵
            :return: 返回神经网络输出
        """
        output = self.layer(x)
        return output


def __MLP_API__(train_X, train_Y, test_X, parameters):
    """
    1、对测试集和训练集进行格式转换(数组->tensor)
       对参数parameters进行解析
    """
    m = np.size(train_X, 1)  # 输入的维度
    x_train = torch.Tensor(train_X)  # Tensor是float类型
    y_train = torch.LongTensor(train_Y)  # LongTensor是Int类型
    x_test = torch.Tensor(test_X)

    '''
    2.加载模型和数据集
    '''
    model = MyModel(input_dim=m, output_dim=parameters["outputdim"], dim1=64, dim2=32)  # 定义模型参数
    print(model)
    loss_fun = nn.CrossEntropyLoss()  # 选择损失函数
    optimizer = optim.Adam(model.parameters(), lr=parameters["lr"])  # 设置优化算法，有SGD AdamW等，参数键doc网站

    # 这样可以看每一层的参数
    # for k, v in model.named_parameters:
    #         print(np.shape(v))

    train_dataset = MyDataset(x_train, y_train)
    # batch_size设置每一批样本集的大小， shuffle=true设置是否打乱顺序
    train_loader = Data.DataLoader(train_dataset, batch_size=parameters["batch_size"], shuffle=True)
    # print(iter(train_loader).__next__())

    '''
    3.开始在模型上进行训练
    '''
    Epoch = parameters["epoch"]  # 总迭代次数
    loss_list = []

    print("\n==========================training==============================\n")
    for epoch in range(Epoch):
        for i, (x, y) in enumerate(train_loader):
            pred = model(x)  # 预测
            loss = loss_fun(pred, y)  # 计算损失

            '''
            三件套
            '''
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 清空计算图

        with torch.no_grad():
            loss_list.append(loss.tolist())
            plt.cla()
            plt.plot(loss_list, 'r-', lw=1)  # 直接输入y轴坐标，不输入x轴坐标是可以的
            plt.ylabel('Loss')
            plt.title('loss=%.4f step=%d' % (loss.cpu(), epoch))
            plt.pause(0.1)

    '''
    4.使用训练好的模型进行泛化测试
    '''
    y_predict = model(x_test)
    print("预测值:")
    print(y_predict)

    # 将输出通过softmax变为概率值
    percentage = torch.softmax(y_predict, dim=1)
    # print(np.shape(output))
    percentage = percentage.detach().numpy()  # Tensor转数组
    percentage = np.round(percentage * 100, 3)  # 乘以100为百分比，保留两位小数
    np.set_printoptions(suppress=True)  # 关闭Numpy的科学计数法
    print("预测标签概率:")
    print(percentage, "%")

    # 选择概率值最大的下标作为预测标签
    # result = np.argmax(y_predict.detach().numpy(), 1)
    result = np.argmax(percentage, 1)
    print("预测标签:")
    print(result)

    if parameters["saveflag"]:
        # 保存网络和训练参数
        date = time.strftime('%m%d_%H%M', time.localtime())  # %Y年份，M月份以此类推
        model_name = date + "_mlp.pth"
        torch.save(model.state_dict(), model_name)  # 保存模型
        print("神经网络保存在", model_name, "文件中")

    return percentage, result
