"""
    @Author : ling chen
    @Create : 2023/11
    @Last modify: 2023/11
    @Description:PYTORCH-cnn接口
"""
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as plt
import numpy as np


def visualize_attention(attention_map, cmap='hot'):
    """
    Visualize an attention map.

    Args:
    - attention_map: A PyTorch tensor of shape (N, 1, H, W) where N is the batch size,
                     H and W are the height and width of the attention map.
    - cmap: The colormap to use for visualization. 'hot' is a good choice for visualizing
            attention maps.
    """
    # 确保是单张图的注意力图
    if attention_map.ndim > 3:
        attention_map = attention_map.squeeze(0)  # 移除批次维度

    # 将注意力图转换为numpy数组，并归一化到0-1范围（如果尚未归一化）
    attention_map_np = attention_map.detach().cpu().numpy()
    attention_map_np = np.clip(attention_map_np, 0, 1)

    # 使用matplotlib显示图像
    plt.imshow(attention_map_np[0, 0], cmap=cmap, interpolation='nearest')
    plt.colorbar()  # 显示颜色条
    plt.axis('off')  # 不显示坐标轴
    plt.show()

class MyDataset(Dataset):
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


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, 1, kernel_size=1, bias=False)

    def forward(self, x):
        # 生成注意力图
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = torch.sigmoid(attention)  # 将注意力值限制在0到1之间

        # 将注意力图与输入特征图相乘
        return x * attention


class CNNWithAttention(nn.Module):
    def __init__(self, H, M,output_dim):
        super(CNNWithAttention, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.attention1 = AttentionBlock(16, 8)  # 注意力块，输出通道数可以调整

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.attention2 = AttentionBlock(32, 16)

        self.fc = nn.Linear(32 * int(H/4) * int(M / 4), output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.attention1(x)  # 应用注意力机制
        x = self.conv2(x)
        x = self.attention2(x)  # 再次应用注意力机制
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

    def get_attention_maps(self, x):
        # 只进行到注意力层，不继续到全连接层
        x = self.conv1(x)
        attention_map1 = self.attention1(x)[:, 0:1, :, :]  # 提取第一个注意力图（假设只关心第一个）

        x = self.conv2(x)
        attention_map2 = self.attention2(x)[:, 0:1, :, :]  # 提取第二个注意力图

        # 如果需要，可以返回两个注意力图，或者只返回你感兴趣的那一个
        return attention_map1, attention_map2

def __AttentionCNN_API_(train_X, train_Y, test_X, parameters):
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
    train_X = torch.Tensor(train_X).unsqueeze(1).to(device)  # 将输入的维度从 (n, a, b) 调整为 (n, 1, a, b)
    train_Y = torch.LongTensor(train_Y).to(device)
    test_X = torch.Tensor(test_X).unsqueeze(1).to(device)
    train_dataset = MyDataset(train_X, train_Y)  # 使用实际的训练集输入和标签
    train_loader = DataLoader(dataset=train_dataset, batch_size=parameters["batch_size"], shuffle=True, drop_last=True)

    '''
    2.加载模型和优化器
    '''
    # 初始化模型和优化器
    model = CNNWithAttention(H, M,parameters["outputdim"]).to(device)  # 模型移到GPU
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

    attention_map1, attention_map2 = model.get_attention_maps(test_X)

    print(f"attention_map2: {attention_map1.shape}")
    # visualize_attention(attention_map1, cmap='hot')
    visualize_attention(attention_map2, cmap='hot')


    y_predict = model(test_X)
    print("预测值:")
    print(y_predict)

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
