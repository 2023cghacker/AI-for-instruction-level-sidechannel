import numpy as np
from matplotlib import pyplot as plt

from src.Configuration_05 import DIRECTORY_PREFIX
from src.Model_MainCode.Loadmatfile import loadCWT_stm32, loaddevice
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        # 使用 ResNet18 模型作为特征提取器
        self.resnet = models.resnet18(pretrained=True)  # 可以选择下载预训练权重
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # 修改输入通道数
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)  # 修改全连接层以匹配类别数

    def forward(self, x):
        return self.resnet(x)

def __ResNet_API__(train_X, train_y, test_X, test_y,  parameters):
    epochs = parameters["epoch"]
    batch_size = parameters["batch_size"]
    num_classes= parameters["outputdim"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    train_X = torch.tensor(train_X, dtype=torch.float32).unsqueeze(1).to(device)  # 转换为 (n, 1, p, q)
    test_X = torch.tensor(test_X, dtype=torch.float32).unsqueeze(1).to(device)
    train_y = torch.tensor(train_y, dtype=torch.long).to(device)
    test_y = torch.tensor(test_y, dtype=torch.long).to(device)

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = ResNetClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练模型
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    # 评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test accuracy: {accuracy:.4f}")

    # 获取预测结果
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())

    return accuracy,predictions

# train_X, train_Y_i, train_Y_d = loadCWT_stm32([1], DIRECTORY_PREFIX + 'DATA_(cwt500s)_{')
# train_X, train_Y_i, train_Y_d = loaddevice([1, 2, 3, 4, 5, 6], 500, DIRECTORY_PREFIX + 'DATA_(200d500s).mat')
# # 创建图形和轴
# # plt.figure(figsize=(10, 5))
# plt.imshow(train_X[0], cmap='viridis', aspect='auto')# 使用 imshow 显示二维数组
# # 添加颜色条
# plt.colorbar()
# plt.show()

# train_X, train_Y_i, train_Y_d = loadCWT_stm32([2,3,4], DIRECTORY_PREFIX + 'DATA_(cwt500s)_{')
# test_X, test_Y_i, test_Y_d = loadCWT_stm32([1], DIRECTORY_PREFIX + 'DATA_(cwt500s)_{')
#
# accuracy,predicted_classes = resnet_classifier(train_X, train_Y_i, test_X, test_Y_i, num_classes=16)
#
# print("Predicted classes:", predicted_classes)
# print("Test accuracy:", accuracy)
