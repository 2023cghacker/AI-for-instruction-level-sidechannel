import time

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Dataset,DataLoader


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


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # LSTM层的输出包括输出序列和(h_n, c_n)，我们只关心最后一个时间步的输出
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        lstm_out = lstm_out[:, -1, :]
        # 通过全连接层进行分类
        out = self.fc(lstm_out)
        return out


def __LSTM_API__(train_X, train_Y, test_X, parameters):
    """
    :param train_X: 训练集输入，类型为数组
    :param train_Y: 训练集标签，类型为数组
    :param test_X: 测试集输入，类型为数组
    :param parameters: 一些神经网络参数
    :return:
    """

    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    1、对数据集进行格式转换(数组->tensor)
      然后加载数据集
    '''
    print(f"\n\n\ntrain_X.shape={train_X.shape},test_X.shape={test_X.shape},device={device}")
    train_X = torch.Tensor(train_X).to(device)  # 将输入的维度从 (n, a, b) 调整为 (n, 1, a, b)
    train_Y = torch.LongTensor(train_Y).to(device)
    train_dataset = MytrainDataset(train_X, train_Y)  # 使用实际的训练集输入和标签
    train_loader = DataLoader(dataset=train_dataset, batch_size=parameters["batch_size"], shuffle=True, drop_last=True)

    test_X = torch.Tensor(test_X).to(device)
    test_dataset = MytestDataset(test_X)  # 使用实际的训练集输入和标签
    test_loader = DataLoader(dataset=test_dataset, batch_size=parameters["batch_size"], shuffle=False, drop_last=False)

    '''
    2.加载模型和优化器
    '''
    model = LSTMModel(input_dim=np.size(train_X, 2), hidden_dim=1024, num_classes=parameters["outputdim"]).to(
        device)  # 定义模型参数
    print("\n", model)
    loss_fun = nn.CrossEntropyLoss()  # 选择损失函数：MSELoss()均方误差， L1Loss()平均绝对误差， CrossEntropyLoss()交叉熵
    optimizer = optim.Adam(model.parameters(), lr=parameters["lr"])  # 设置优化算法，有SGD AdamW等，参数键doc网站
    # 这样可以看每一层的参数
    # for k, v in model.named_parameters:
    #         print(np.shape(v))

    '''
    3.开始在模型上进行训练
    '''
    Epoch = parameters["epoch"]  # 总迭代次数
    loss_list = []
    total_step = len(train_loader)

    print("\n==========================training==============================\n")
    for epoch in range(Epoch):
        for i, (x, y) in enumerate(train_loader):
            # print(x.shape, y.shape)
            pred = model(x)  # 预测
            loss = loss_fun(pred, y)  # 计算损失
            # if i % 1000 == 0:
            #     print(loss)

            '''
            三件套
            '''
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 清空计算图

            # 清理无用的中间变量
            del x, y, pred
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
    # result = np.argmax(y_predict.detach().numpy(), 1)
    result = np.argmax(percentage, 1)
    print("预测标签:")
    print(result)

    if parameters["saveflag"]:
        # 保存网络和训练参数
        date = time.strftime('%m%d_%H%M', time.localtime())  # %Y年份，M月份以此类推
        model_name = date + "_bpnn.pth"
        model_name = "device1_bpnn.pth"
        torch.save(model.state_dict(), model_name)  # 保存模型
        print("神经网络保存在", model_name, "文件中")

    return percentage, result


# # 假设的输入数据维度
# sequence_length = 10
# input_dim = 8
# hidden_dim = 50
# num_classes = 16
# batch_size = 32
#
# # 生成随机数据
# np.random.seed(0)
# X = np.random.random((batch_size, sequence_length, input_dim))
# # 转换为torch张量
# X = torch.tensor(X, dtype=torch.float32)
#
# y = np.random.randint(num_classes, size=(batch_size,))
# y = torch.tensor(y, dtype=torch.long)  # PyTorch中的类别标签通常是long类型
#
# model = LSTMModel(input_dim, hidden_dim, num_classes)
#
# # 损失函数和优化器
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # 训练模型
# num_epochs = 10
# for epoch in range(num_epochs):
#     optimizer.zero_grad()
#     outputs = model(X)
#     loss = criterion(outputs, y)
#     loss.backward()
#     optimizer.step()
#
#     if (epoch + 1) % 2 == 0:
#         print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
#     # 预测新数据（示例）
# new_samples = torch.tensor(np.random.random((5, sequence_length, input_dim)), dtype=torch.float32)
# predictions = model(new_samples)
# _, predicted = torch.max(predictions.data, 1)
# print("Predictions:", predicted.tolist())
