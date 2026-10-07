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

# 模型参数
D_MODEL = 96 # 每个 Token 的向量维度
HEADS = 6 # 注意力头个数，每个头 16 维
D_FF = 256 # SwiGLU 前馈网络的中间层维度
NUM_LAYERS = 4 # GPT Block 的层数
MAX_LENGTH = 32 # 最大上下文长度，需大于本例完整序列长度 26
ROPE_BASE = 1000.0 # RoPE 的频率基数

MIN_TOKEN_FREQ = 5
character_counts = Counter("".join(all_poems))
kept_characters = {
    char for char, count in character_counts.items() if char in "，。" or count >= MIN_TOKEN_FREQ
}

tokens = ["<BOS>", "<EOS>", "<UNK>"] + sorted(kept_characters)
token_ids = { token: index for index, token in enumerate(tokens) }
BOS = token_ids["<BOS>"]
EOS = token_ids["<EOS>"]
UNK = token_ids["<UNK>"]

# 训练时随机屏蔽 10% 的 Embedding 输出特征
DROPOUT = 0.1

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # 每个 Token 对应参数表中的一行，向量维度由 D_MODEL 决定
        self.embedding = nn.Embedding(vocab_size, D_MODEL)

        # 后续输出投影复用这张表；小幅初始化可避免初始 logits 过大
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # train() 模式启用；调用 eval() 后关闭，用于评估和生成
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, token_ids):
        # 查表后施加 Dropout; 这里不叠加位置向量，也不乘 sqrt(D_MODEL)
        # 位置信息留给后续注意力层处理
        return self.dropout(self.embedding(token_ids))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim, max_length, base=ROPE_BASE):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE 要求每个注意力头的维度维偶数。"

        # 每一对维度共享一个频率 theta_i = base^(-2i/head_dim)。
        # head_dim 个特征按 (0, 1)、(2, 3)... 两两组成二维平面，因此只需 head_dim / 2 个频率
        pair_index = torch.arange(0, head_dim // 2, dtype=torch.float32)

        # 低维频率高、旋转快; 高维频率低、旋转慢，从而覆盖不同尺度的位置关系
        theta = base ** (-2.0 * pair_index / head_dim) # [head_dim/2]
        position = torch.arange(max_length, dtype=torch.float32) # [L]
        angles = torch.outer(position, theta) # [L, head_dim/2]

        # 预存每个位置、每对维度对应旋转角的余弦与正弦
        # cos/sin 不参与训练；register_buffer 让它们随模型迁移到 CPU、CUDA 或者 MPS
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x):
        # x: [B, H, L, head_dim], 按相邻维度两两配对旋转
        sequence_length = x.shape[-2]

        # cos/sin 的 [L, head_dim/2] 会自动广播到 Batch 维与 Head 维
        cos = self.cos[:sequence_length]
        sin = self.cos[:sequence_length]
        even = x[..., 0::2] # 偶数维 x_2j
        odd = x[..., 1::2] # 奇数维 x_2j+1
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        # 交错合并会原来的维度顺序
        # stack 后为[..., head_dim/2, 2], flatten 将每对维度交错还原为 [..., head_dim]
        rottated = torch.stack((rotated_even, rotated_odd), dim=-1)
        return rottated.flatten(-2)

def attention(q, k, v):
    # 单头 [L, Dh] 与批量多头 [B, H, L, Dh] 共用同一计算，@ 作用与最后两维
    # 转置 K 的最后两维后，Q @ K^T 得到 [..., L, L] 的注意力分数
    # 除以 sqrt(Dh) 可控制点积数值范围，避免 softmax 过早进入接近 one-hot 的饱和区
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])

    sequence_length = q.shape[-2]
    # diagonal=1 只遮住严格位于主对角线右上方的位置：第 i 个 Token 可看见自己和全部前文
    # mask 与 q 放在同一设备，避免在 GPU/MPS 训练时发生设备不一致的问题
    mask = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=q.device), diagonal=1)

    # [L, L] 掩码自动广播到每个 Batch、每个 Head 的分数矩阵
    scores = scores.masked_fill(mask, float("-inf"))

    # softmax 在 Key 维度归一化，使每个 Query 对所有可见位置的权重之和为 1
    weights = torch.softmax(scores, dim=-1)
    # [B, H, L, L] @ [B, H, L, Dh] -> [B, H, L, Dh]
    return weights @ v
