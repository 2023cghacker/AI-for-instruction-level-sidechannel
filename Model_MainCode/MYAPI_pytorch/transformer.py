import torch
import torch.nn as nn
import math
from torchviz import make_dot


class MyTransformer(nn.Module):
    """
    继承父类 torch.nn.Module
    """

    def __init__(self, input_dim, model_dim, num_heads, num_layers, num_classes):
        super(MyTransformer, self).__init__()

        # 定义嵌入层
        self.embedding = nn.Linear(input_dim, model_dim)

        # 定义位置编码
        self.position_encoding = PositionalEncoding(model_dim)

        # 定义多个自注意力层和前馈神经网络层
        self.layers = nn.ModuleList([
            TransformerLayer(model_dim, num_heads) for _ in range(num_layers)
        ])

        # 定义输出层
        self.fc = nn.Linear(model_dim, num_classes)

    def forward(self, x):
        """
        前向传播
        :param x: 输入矩阵，形状为 (batch_size, sequence_length, input_dim)
        :return: 返回神经网络输出
        """
        # 嵌入和位置编码
        x = self.embedding(x)
        x = self.position_encoding(x)

        # 通过多个Transformer层
        for layer in self.layers:
            x = layer(x)

        # 聚合时间步并通过全连接层输出分类结果
        x = x.mean(dim=1)
        output = self.fc(x)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % num_heads == 0
        self.d_k = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_layers = nn.ModuleList([nn.Linear(model_dim, model_dim) for _ in range(3)])
        self.fc = nn.Linear(model_dim, model_dim)

        self.attention = None

    def forward(self, query, key, value):
        batch_size = query.size(0)

        query, key, value = [
            l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (query, key, value))
        ]

        x, self.attention = self.scaled_dot_product_attention(query, key, value)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.fc(x)

    def scaled_dot_product_attention(self, query, key, value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = scores.softmax(dim=-1)
        return torch.matmul(p_attn, value), p_attn


class TransformerLayer(nn.Module):
    def __init__(self, model_dim, num_heads):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(),
            nn.Linear(model_dim * 4, model_dim)
        )
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        attn_output = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.layer_norm1(x)
        ff_output = self.feed_forward(x)
        x = x + ff_output
        x = self.layer_norm2(x)
        return x


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 假设输入数据的形状为 (batch_size, sequence_length, feature_dim)
data = torch.randn(100, 10, 16)
labels = torch.randint(0, 2, (100,))

dataset = TimeSeriesDataset(data, labels)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

model = MyTransformer(input_dim=16, model_dim=32, num_heads=4, num_layers=2, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 创建输入样本
sample_data = next(iter(dataloader))[0]

# 生成计算图
y = model(sample_data)
make_dot(y, params=dict(model.named_parameters())).render("transformer", format="png")

# 训练循环
for epoch in range(10):
    for batch_data, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
