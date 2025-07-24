"""
    @Author : ling chen
    @Create : 2023/07
    @Last modify: 2023/07
    @Description: Pytorch的使用教程模板文件，算法内容是对数字进行判断是否能被3，5，15整除，
                  输入是二进制数据，输出是0，1，2，3 四个标签
"""

import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim

# 设置GPU加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fizzbuzz_encode(number):
    """
    :param number: 数字
    :return: 返回标签
    """
    if number % 15 == 0:
        return 3
    elif number % 5 == 0:
        return 2
    elif number % 3 == 0:
        return 1
    return 0


def fizzbuzz_decode(number, label):
    """
    :param number: 数字
    :param label: 标签
    :return: 返回中文名称
    """
    return ['不能被3or5整除', '能被3整除', '能被5整除', '能被15整除'][label]


def test():
    """
    @测试一下上面定义的两个函数
    """
    for number in range(1, 16):
        print(fizzbuzz_decode(number, fizzbuzz_encode(number)))


test()


def binary_encode(number):
    """
    :param number: 数字（十进制）
    :return: 10位的二进制数字（一维数组形式）
    """
    return np.array([number >> d & 1 for d in range(10)][::-1])


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

    def __init__(self, dim1, dim2):
        super(MyModel, self).__init__()
        self.activation = nn.ReLU()
        # 自己定义了3层网络，这里可以选择其他的算法
        self.linear1 = nn.Linear(10, dim1)  # (输入维度，输出维度)
        self.linear2 = nn.Linear(dim1, dim2)
        self.linear3 = nn.Linear(dim2, 4)  # 4是最终输出标签的总数目

    def forward(self, x):
        '''
        :param x: 输入矩阵
        :return: 返回输出（预测值）
        '''
        output = self.linear1(x)  # => [batchsize,64]
        output = self.activation(output)  # => [batchsize,64]
        output = self.linear2(output)  # => [batchsize,8]
        output = self.linear3(output)  # => [batchsize,4]
        return output


'''
1.创建训练集
'''
# binary_encode(number) 将数字编码成二进制
x_train = torch.Tensor([binary_encode(number) for number in range(101, 1024)])  # Tensor是float类型
print(x_train)
print("训练集数据的大小为", np.shape(x_train))
# fizzbuzz_encode(number)得到每个数字（输入）的标签
y_train = torch.LongTensor([fizzbuzz_encode(number) for number in range(101, 1024)])  # LongTensor是Int类型
print(y_train)
print("训练集标签的大小为", np.shape(y_train))

'''
2.加载模型和数据集
'''
model = MyModel(64, 8).to(device)  # 定义模型
loss_fun = nn.CrossEntropyLoss()  # 选择损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # 设置优化算法，有SGD AdamW等，参数键doc网站

# 这样可以看每一层的参数
for k, v in model.named_parameters():
    print(k)
    if k == 'linear1.weight':
        # print(v)
        print(np.shape(v))

train_dataset = MyDataset(x_train, y_train)
train_loader = Data.DataLoader(train_dataset, batch_size=16, shuffle=True)  # batch_size设置每一批样本集的大小， shuffle=true设置打乱顺序
print(iter(train_loader).__next__())

'''
3.开始在模型上进行训练
'''
Epoch = 30  # 总迭代次数
for epoch in range(Epoch):
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        pred = model(x)  # 预测
        loss = loss_fun(pred, y)  # 计算损失
        if i % 10 == 0:
            print(loss)

        '''
        三件套
        '''
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        optimizer.zero_grad()  # 清空计算图

'''
4.构造测试样本（集）进行泛化测试
'''
x_test = torch.Tensor([binary_encode(15)]).to(device)
y_predict = model(x_test)
print("预测值")
print(y_predict)

# 将输出通过softmax变为概率值
output = torch.softmax(y_predict, dim=1)
output = output[0].detach().numpy()  # Tensor转数组
output = np.round(output * 100, 3)  # 乘以100为百分比，保留两位小数
np.set_printoptions(suppress=True)  # 关闭Numpy的科学计数法
print("预测标签概率")
print(output, "%")

result = np.argmax(y_predict.detach().numpy(), 1)
print("预测标签")
print(result)

'''
5.附录：数值转换方法
'''
a = [1, 2, 3, 5, 8, 9.3]


def exchange(a):
    # 输入一个列表
    print(type(a))
    array = np.array(a)  # list转array
    print(type(array))
    list = array.tolist()  # array转list
    print(type(list))
    tensor = torch.from_numpy(array)  # array转tensor
    print(type(tensor))
    array = tensor.numpy()  # tensor转array
    print(type(array))


exchange(a)
