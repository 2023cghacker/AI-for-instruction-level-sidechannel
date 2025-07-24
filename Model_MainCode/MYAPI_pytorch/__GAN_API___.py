import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import src.Configuration_matplot
from src.Configuration_01 import DIRECTORY_PREFIX
from src.Configuration_02 import DIRECTORY_PREFIX as addprefix
from src.Model_MainCode.Loadmatfile import loadData, extract_name


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


# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_size),
            # nn.Identity()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, output_size),
            # nn.Sigmoid()
            nn.Softmax()
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x


def __GAN_API__(source_X, source_Y, target_X, target_Y):
    """
    :param source_X: 源数据集
    :param target_X: 目标数据集
    :return: 基于源数据的维度生成目标数据的维度
    """
    '''
    1、对数据集进行格式转换(数组->tensor)
      然后加载数据集到Loader
    '''
    # 源数据集(低维)
    source_X = torch.Tensor(source_X)
    source_Y = torch.LongTensor(source_Y)
    # source_Y = torch.LongTensor(np.zeros((len(source_X))))
    the_dataset = MyDataset(source_X, source_Y)

    # 目标数据集(高维)
    target_X = torch.Tensor(target_X)
    target_Y = torch.LongTensor(target_Y)
    # target_Y = torch.LongTensor(np.ones((len(target_X))))
    the_dataset = MyDataset(target_X, target_Y)
    high_dataloader = DataLoader(dataset=the_dataset, batch_size=parameters["batch_size"], shuffle=True)

    '''
    2、创建生成器G, 判别器D, 生成器损失 (LossG), 判别器损失 (LossD)
    生成器：从低维生成高维数据
    判别器：对生成的数据(标签=0)和真实的数据(标签=1)进行区分
    '''
    G = Generator(input_size=np.size(source_X, 1), output_size=np.size(target_X, 1))  # 生成器
    D = Discriminator(input_size=np.size(target_X, 1), output_size=42)  # 判别器
    print(G, '\n\n', D)

    # 定义GAN损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # BCELoss()  # 二元交叉熵损失函数
    optimizer_g = optim.Adam(G.parameters(), lr=parameters["lr"])
    optimizer_d = optim.Adam(D.parameters(), lr=parameters["lr"])
    num_epochs = parameters["Epoch"]  # 运行总次数
    lossG_list = []
    lossD_list = []

    plt.figure(figsize=(14, 8))
    for epoch in range(num_epochs):
        for (real_data, real_labels) in high_dataloader:
            '''
            训练判别器
            '''
            output_real = D(real_data)  # 判别
            loss_real = criterion(output_real, real_labels)

            fake_data = G(source_X)  # 生成fake数据
            fake_labels = source_Y + 21  # fake数据的标签
            output_fake = D(fake_data.detach())
            loss_fake = criterion(output_fake, fake_labels)

            loss_d = loss_fake + loss_real  # 判别器损失 (LossD)越小代表越能区分出虚假数据
            loss_d.backward()
            optimizer_d.step()
            optimizer_d.zero_grad()

            '''
              训练生成器
            '''
            fake_data = G(source_X)  # next(iter(low_dataloader))[0])  # 生成虚假数据
            output_fake = D(fake_data)  # 判别器对虚假数据进行识别标签
            true_labels = source_Y
            # true_labels = torch.LongTensor(np.ones((len(fake_data))))
            loss_g = criterion(output_fake, true_labels)  # 生成器损失 (LossG)越小代表生成的数据越接近真实数据
            loss_g.backward()
            optimizer_g.step()
            optimizer_g.zero_grad()

        with torch.no_grad():
            lossG_list.append(loss_g.tolist())
            lossD_list.append(loss_d.tolist())
            '''
            绘图
            '''
            plt.clf()
            plt.subplot(3, 2, 1)
            plt.plot(source_X[0])
            plt.title("源能量迹 ")  # Dimension=%d" % (len(source_X[0])), fontsize=12)
            plt.subplot(3, 2, 3)
            plt.plot(target_X[0])
            plt.title("目标能量迹 ")  # Dimension=%d" % (len(target_X[0])), fontsize=12)
            plt.subplot(3, 2, 5)
            plt.plot(fake_data[0])
            plt.title("GAN生成的迁移能量迹", fontsize=12)
            plt.subplot(1, 2, 2)
            plt.plot(lossG_list, lw=2)
            plt.plot(lossD_list, lw=2)
            plt.ylabel('Loss')
            plt.title('lossG=%.4f lossD=%.4f Epoch=%d' % (loss_g.cpu(), loss_d.cpu(), epoch))
            plt.tight_layout()
            plt.pause(0.01)

        if (epoch + 1) % 5 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Loss fake: {loss_fake.item():.4f}, Loss real: {loss_real.item():.4f}')

    # 使用生成器生成高维数据集
    fake_X = G(source_X)
    generated_data = fake_X.detach().numpy()
    print(generated_data.shape)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(source_X[0])
    plt.title("源能量迹")
    plt.subplot(3, 1, 2)
    plt.plot(target_X[0])
    plt.title("目标能量迹")
    plt.subplot(3, 1, 3)
    plt.plot(generated_data[0])
    plt.title("GAN生成的迁移能量迹")
    plt.tight_layout()
    plt.show()

    # 使用判别器进行识别
    outputs = D(fake_X)
    print("预测值:\n", outputs)
    # 将输出通过softmax变为概率值
    percentage = torch.softmax(outputs, dim=1)
    percentage = percentage.detach().numpy()  # Tensor转数组
    percentage = np.round(percentage * 100, 3)  # 乘以100为百分比，保留两位小数
    np.set_printoptions(suppress=True)  # 关闭Numpy的科学计数法
    print("预测标签概率:\n", percentage, "%")
    # 选择概率值最大的下标作为预测标签
    result = np.argmax(percentage, 1)
    print("预测标签:\n", result, "\n预测为真实的占比:", sum(result) / len(result))

    outputs = D(target_X)
    print("预测值:\n", outputs)
    # 将输出通过softmax变为概率值
    percentage = torch.softmax(outputs, dim=1)
    percentage = percentage.detach().numpy()  # Tensor转数组
    percentage = np.round(percentage * 100, 3)  # 乘以100为百分比，保留两位小数
    np.set_printoptions(suppress=True)  # 关闭Numpy的科学计数法
    print("预测标签概率:\n", percentage, "%")
    # 选择概率值最大的下标作为预测标签
    result = np.argmax(percentage, 1)
    print("预测标签:\n", result, "\n预测为真实的占比:", sum(result) / len(result))
    print(result)
    return generated_data


if __name__ == '__main__':
    source_X, source_Y = loadData('../../DataFile/Datasets_Small/DATA_(42d100s).mat')
    target_X, target_Y = loadData('../../DataFile/Datasets_Big/DATA_(500d500s).mat')
    target_X = target_X[0:500, :]
    target_Y = target_Y[0:500]
    parameters = {"batch_size": 100, "lr": 1e-3, "Epoch": 100}

    generated_data = __GAN_API__(source_X, source_Y, target_X, target_Y)
    # 保存
    DATAName = "../../DataFile/Datasets_Big/DATA_(GAN500d500s).mat"
    scipy.io.savemat(DATAName, {'X': generated_data, 'Y': source_Y})
    print("训练集测试集已经保存在：", extract_name(DATAName), "\n")
