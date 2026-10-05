import math
import random
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn

# 固定 Python 与 PyTorch 的随机种子，使数据顺序、参数初始化和演示可复现
random.seed(42)
torch.manual_seed(42)

# 小模型的单次矩阵运算很短，限制 CPU 线程可避免线程调度开销反而拖慢运行
torch.set_num_threads(1)

# 后续模型参数和训练 Batch 必须移动到同一个 DEVICE, 否则张量运算会报设备不一致
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print("训练设备：", DEVICE)


