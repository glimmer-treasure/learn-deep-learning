import random
import torch

# 固定随机种子以确保结果可复现
random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(1)

def main() -> None:
    print("Hello from mini-transformer!")
