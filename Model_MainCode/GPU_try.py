import torch
from torch import nn
#  查看gpu信息
cudaMsg = torch.cuda.is_available()
gpuCount = torch.cuda.device_count()
print("是否存在GPU:{}".format(cudaMsg), "\n如果存在有：{}个".format(gpuCount))
print(torch.__version__)
print(torch.version.cuda)