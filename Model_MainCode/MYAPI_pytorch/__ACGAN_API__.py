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
        self.fc1 = nn.Linear(input_size, 16)

        self.t1 = nn.Sequential(
            nn.Linear(16, 32),
            nn.Tanh(),
        )
        self.t2 = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
        )
        self.t3 = nn.Sequential(
            nn.Linear(16, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 16)
        x = self.t1(x)
        x = self.t2(x)
        x = self.t3(x)
        return x


# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_size, classesnum):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
        )
        self.fc_source = nn.Linear(16, 1)
        self.fc_class = nn.Linear(16, classesnum)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.layer(x)
        x = x.view(-1, 16)
        rf = self.sig(self.fc_source(x))  # data generated(fake) or from training set(real)
        c = self.soft(self.fc_class(x))
        return rf, c


def __ACGAN_API__(raw_X, raw_Y, target_X, target_Y):
    """
    :param raw_X: 源数据集
    :param target_X: 目标数据集
    :return: 基于源数据的维度生成目标数据的维度
    """
    '''
    1、对数据集进行格式转换(数组->tensor)
      然后加载数据集到Loader
    '''
    # 源数据集(低维)
    raw_X = torch.Tensor(raw_X)
    raw_Y = torch.LongTensor(raw_Y)

    # 目标数据集(高维)
    target_X = torch.Tensor(target_X)
    target_Y = torch.LongTensor(target_Y)
    the_dataset = MyDataset(target_X, target_Y)
    batchsize = parameters["batch_size"]
    train_loader = DataLoader(dataset=the_dataset, batch_size=batchsize, shuffle=True)
    '''
    2、创建生成器G, 判别器D, 生成器损失 (LossG), 判别器损失 (LossD)
    生成器：从低维生成高维数据
    判别器：对生成的数据(标签=0)和真实的数据(标签=1)进行区分
    '''
    G = Generator(input_size=np.size(raw_X, 1), output_size=np.size(target_X, 1))  # 生成器
    D = Discriminator(input_size=np.size(target_X, 1), classesnum=21)  # 判别器
    print(G, '\n\n', D)

    # 定义GAN损失函数和优化器
    source_obj = nn.BCELoss()  # source-loss
    class_obj = nn.NLLLoss()  # class-loss

    optimizer_g = optim.Adam(G.parameters(), lr=parameters["lr"])
    optimizer_d = optim.Adam(D.parameters(), lr=parameters["lr"])
    lossG_list = []
    lossD_list = []

    plt.figure(figsize=(14, 8))
    num_epochs = parameters["Epoch"]  # 运行总次数
    for epoch in range(num_epochs):
        for (inputs, outputs) in train_loader:
            '''
            训练判别器
            '''
            # training with real data----
            source_, class_ = D(inputs)  # 判别器给出两个标签，一个是真假标签，一个是分类标签
            source_ = torch.reshape(source_, [-1])
            real_label = torch.FloatTensor(batchsize)
            real_label.fill_(1)  # 真标签
            source_error = source_obj(source_, real_label)  # 真假的损失函数
            class_error = class_obj(class_, outputs)  # 类别的损失函数

            error_real = source_error + class_error
            # error_real.backward()
            # optimizer_d.step()  # 判别器的优化器迭代

            # training with fake data now----
            fake_data = G(raw_X)  # 生成fake数据
            fake_output = raw_Y

            source_, class_ = D(fake_data.detach())
            source_ = torch.reshape(source_, [-1])
            fake_label = torch.FloatTensor(len(raw_Y))
            fake_label.fill_(0)  # 假标签
            source_error = source_obj(source_, fake_label)
            class_error = class_obj(class_, fake_output)

            error_fake = source_error + class_error
            # error_fake.backward()

            loss_d = error_fake + error_real  # 判别器损失 (LossD)越小代表越能区分出虚假数据
            loss_d.backward()  # 反向传播
            optimizer_d.step()  # 判别器更新参数
            optimizer_d.zero_grad()  # 清空梯度

            '''
              训练生成器
            '''
            # G.zero_grad()

            source_, class_ = D(fake_data)  # 判别器对虚假数据进行识别标签
            source_ = torch.reshape(source_, [-1])

            real_label = torch.FloatTensor(len(raw_Y))
            real_label.fill_(1)  # 真标签
            source_error = source_obj(source_, real_label)
            class_error = class_obj(class_, fake_output)
            loss_g = source_error + class_error
            loss_g.backward()  # 反向传播
            optimizer_g.step()  # 生成器更新参数
            optimizer_g.zero_grad()  # 清空梯度

        with torch.no_grad():
            lossG_list.append(loss_g.tolist())
            lossD_list.append(loss_d.tolist())
            '''
            绘图
            '''
            plt.clf()
            plt.subplot(3, 2, 1)
            plt.plot(raw_X[0])
            plt.title("源能量迹 ")  # Dimension=%d" % (len(raw_X[0])), fontsize=12)
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
                f'Epoch [{epoch + 1}/{num_epochs}], Loss fake: {error_fake.item():.4f}, Loss real: {error_real.item():.4f}')

    # 使用生成器生成高维数据集
    fake_X = G(raw_X)
    generated_data = fake_X.detach().numpy()
    print(generated_data.shape)

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(raw_X[0])
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
    source_, class_ = D(fake_X)
    print("预测值:\n")
    print("source_:\n", source_)
    print("class_:\n", class_)
    # 将输出通过softmax变为概率值
    # percentage = torch.softmax(outputs, dim=1)
    # percentage = percentage.detach().numpy()  # Tensor转数组
    # percentage = np.round(percentage * 100, 3)  # 乘以100为百分比，保留两位小数
    # np.set_printoptions(suppress=True)  # 关闭Numpy的科学计数法
    # print("预测标签概率:\n", percentage, "%")
    # # 选择概率值最大的下标作为预测标签
    # result = np.argmax(percentage, 1)
    # print("预测标签:\n", result, "\n预测为真实的占比:", sum(result) / len(result))
    #
    # outputs = D(target_X)
    # print("预测值:\n", outputs)
    # # 将输出通过softmax变为概率值
    # percentage = torch.softmax(outputs, dim=1)
    # percentage = percentage.detach().numpy()  # Tensor转数组
    # percentage = np.round(percentage * 100, 3)  # 乘以100为百分比，保留两位小数
    # np.set_printoptions(suppress=True)  # 关闭Numpy的科学计数法
    # print("预测标签概率:\n", percentage, "%")
    # # 选择概率值最大的下标作为预测标签
    # result = np.argmax(percentage, 1)
    # print("预测标签:\n", result, "\n预测为真实的占比:", sum(result) / len(result))
    # print(result)
    return generated_data


if __name__ == '__main__':
    raw_X, raw_Y = loadData('../../DataFile/Datasets_Small/DATA_(42d100s).mat')
    target_X, target_Y = loadData('../../DataFile/Datasets_Big/DATA_(500d500s).mat')
    target_X = target_X[0:500, :]
    target_Y = target_Y[0:500]
    parameters = {"batch_size": 100, "lr": 1e-3, "Epoch": 100}

    generated_data = __ACGAN_API__(raw_X, raw_Y, target_X, target_Y)
    # 保存
    DATAName = "../../DataFile/Datasets_Big/DATA_(GAN500d500s).mat"
    scipy.io.savemat(DATAName, {'X': generated_data, 'Y': raw_Y})
    print("训练集测试集已经保存在：", extract_name(DATAName), "\n")
