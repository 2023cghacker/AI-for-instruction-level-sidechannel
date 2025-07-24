"""
    @Author : ling chen
    @Create : 2023/10
    @Last modify: 2023/11
    @Description:PYTORCH-bpnn接口
"""
import time
from torchviz import make_dot
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


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 添加一个伪序列维度
        x = x.unsqueeze(1)  # 形状变为 [batch_size, 1, feature_dim]

        Q = self.query(x)  # 形状为 [batch_size, 1, in_dim]
        K = self.key(x)  # 形状为 [batch_size, 1, in_dim]
        V = self.value(x)  # 形状为 [batch_size, 1, in_dim]

        # 计算注意力分数，形状为 [batch_size, 1, 1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention_weights = self.softmax(scores)  # 形状为 [batch_size, 1, 1]
        context = torch.matmul(attention_weights, V)  # 形状为 [batch_size, 1, in_dim]

        # 移除伪序列维度
        context = context.squeeze(1)  # 形状变为 [batch_size, in_dim]

        return context


class AttentionDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionDNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # 第一层隐藏层
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.bn1 = nn.BatchNorm1d(256)  # 第一层批归一化
        self.attention1 = SelfAttention(256)  # 自注意力层
        self.fc2 = nn.Linear(256, 128)  # 第二层隐藏层
        self.relu2 = nn.ReLU()  # ReLU激活函数
        self.bn2 = nn.BatchNorm1d(128)  # 第二层批归一化
        self.attention2 = SelfAttention(128)  # 自注意力层
        self.output = nn.Linear(128, output_dim)  # 输出层

    def forward(self, x):
        x = self.fc1(x)  # 通过第一层隐藏层
        x = self.relu1(x)  # 激活函数
        x = self.bn1(x)  # 批归一化
        x = self.attention1(x)  # 自注意力层
        x = self.fc2(x)  # 通过第二层隐藏层
        x = self.relu2(x)  # 激活函数
        x = self.bn2(x)  # 批归一化
        x = self.attention2(x)  # 自注意力层
        x = self.output(x)  # 输出层
        return x

def __AttentionDNN_API__(train_X, train_Y, test_X, parameters):
    """
    :param train_X: 训练集输入，类型为数组
    :param train_Y: 训练集标签，类型为数组
    :param test_X: 测试集输入，类型为数组
    :param parameters: 一些神经网络参数
    :return:
    """

    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    '''
    1、对数据集进行格式转换(数组->tensor)
      然后加载数据集
    '''
    print(f"train_X.shape={train_X.shape},test_X.shape={test_X.shape}")
    x_train = torch.Tensor(train_X).to(device)  # Tensor是float类型
    y_train = torch.LongTensor(train_Y).to(device)  # LongTensor是Int类型
    x_test = torch.Tensor(test_X).to(device)
    train_dataset = MyDataset(x_train, y_train)
    # batch_size设置每一批样本集的大小， shuffle=true设置是否打乱顺序
    train_loader = Data.DataLoader(train_dataset, batch_size=parameters["batch_size"], shuffle=True)

    '''
    2.加载模型和优化器
    '''
    model = AttentionDNN(input_dim=np.size(train_X, 1), output_dim=parameters["outputdim"]).to(device)  # 定义模型参数
    print("\n", model)
    loss_fun = nn.CrossEntropyLoss()  # 选择损失函数：MSELoss()均方误差， L1Loss()平均绝对误差， CrossEntropyLoss()交叉熵
    optimizer = optim.Adam(model.parameters(), lr=parameters["lr"])  # 设置优化算法，有SGD AdamW等，参数键doc网站
    # 这样可以看每一层的参数
    # for k, v in model.named_parameters:
    #         print(np.shape(v))

    # 生成计算图
    # y = model(torch.Tensor(train_X).to(device))
    # make_dot(y, params=dict(model.named_parameters())).render("attentionDnn", format="png")

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
            # if i % 1000 == 0:
            #     print(loss)

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
            plt.title('loss=%.4f 当前迭代次数=%d' % (loss.cpu(), epoch))
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
        torch.save(model.state_dict(), model_name)  # 保存模型
        print("神经网络保存在", model_name, "文件中")

    return percentage, result
