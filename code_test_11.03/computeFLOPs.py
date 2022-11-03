"""
使用fvcore计算Pytorch中模型的参数数量以及FLOPs
安装fvcore：pip install fvcore
"""

import torch
from torchvision.models import resnet50
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# 创建resnet50网络
model = resnet50(num_classes=1000)

# 创建输入网络的tensor
tensor = (torch.rand(1, 3, 224, 224),)

# 分析FLOPs
flops = FlopCountAnalysis(model, tensor)
print("FLOPs: ", flops.total())

# 分析parameters
print(parameter_count_table(model))
