import torch
from torch import nn
from .config import D, HEADS, D_FF, NUM_LAYERS, MAX_LENGTH
from .dataset import source_words, target_words
import math

def attention(q, k, v, causal=False):
    # [Lq, Dh] @ [Dh, Lk] -> [Lq, Lk], 每行是一个 Query 的分数
    scores = q @ k.T / math.sqrt(q.shape[-1])

    if causal:
        # 生成一个上三角矩阵，形状为 [Lq, Lk]，上三角部分为 -inf，下三角部分为 0
        mask = torch.triu(torch.ones_like(scores, dtype = torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))

    # 沿 Key 的方向归一化；被屏蔽的位置的权重为 0
    weights = torch.softmax(scores, dim=-1) # [Lq, Lk]
    return weights @ v # [Lq, Lk] @ [Lk, Dh] -> [Lq, Dh]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = MAX_LENGTH) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(d_model // 2):
                angle = pos / (10000.0 ** (2 * i / d_model))
                pe[pos, 2 * i] = math.sin(angle)
                pe[pos, 2 * i + 1] = math.cos(angle)
        # 固定位置表，不参与梯度更新（注册为 buffer，这样它不会被视为模型参数，但会随模型保存和加载）
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows = []
        for pos in range(len(x)):
            rows.append(x[pos] + self.pe[pos]) # 两个 [D] 向量逐元素相加
        return torch.stack(rows) # 把多行 [D] 叠成 [句长, D] 的张量

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # 创建形状为 [](vocab_size, D) 的词向量表，每个 Token ID 对应一行
        self.embedding = nn.Embedding(vocab_size, D)
        self.position = PositionalEncoding(D, MAX_LENGTH)

    def forward(self, token_ids) -> torch.Tensor:
        # 输入为一维 Token ID 张量，形状为 [L]
        # 按 ID 查出每个 Token 的词向量，得到[L, D], 再按论文乘以 sqrt(D) 进行缩放
        x = self.embedding(token_ids) * math.sqrt(D)
        return self.position(x) # 加入位置编码，输出形状仍未 [L, D] 的张量

class AttentionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        head_dim = D // HEADS # 单个头维度 Dh
        # 分别学习Q、K、V 的投影，将每个位置从 D 维映射到 Dh 维
        self.q_proj = nn.Linear(D, head_dim)
        self.k_proj = nn.Linear(D, head_dim)
        self.v_proj = nn.Linear(D, head_dim)

    def forward(self, query, key, value, causal=False) -> torch.Tensor:
        q = self.q_proj(query) # [Lq, D] -> [Lq, Dh]
        k = self.k_proj(key)   # [Lk, D] -> [Lk, Dh]
        v = self.v_proj(value) # [Lv, D] -> [Lv, Dh]
        return attention(q, k, v, causal)

class MultiHeadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 每个头分别学习自己的 Q、K、V 投影，参数相互独立
        self.heads = nn.ModuleList([AttentionHead() for _ in range(HEADS)])
        # 对拼接后的特征做输出投影，融合各个头的信息，维度保持 D
        self.out_proj = nn.Linear(D, D) # 将多头的输出再映射回 D 维

    def forward(self, query, key, value, causal=False) -> torch.Tensor:
        outputs = []
        # 所有头读取相同输入，各自得到一个 [Lq, Dh] 的输出
        for head in self.heads:
            outputs.append(head(query, key, value, causal)) # 每个头输出 [Lq, Dh]
        # 沿特征维拼接：HEADS 个 [Lq, Dh] -> [Lq, HEADS * Dh] = [Lq, D]
        joined = torch.cat(outputs, dim=-1) # [Lq, D]
        return self.out_proj(joined) # 输出 [Lq, D]

class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 两层全连接网络，中间用 ReLU 激活函数
        self.network = nn.Sequential(
            nn.Linear(D, D_FF),
            nn.ReLU(),
            nn.Linear(D_FF, D)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x) # [L, D] -> [L, D]

class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_mha = MultiHeadAttention() # 多头自注意力
        self.ffn = FeedForward() # FNN: 逐位置加工特征
        self.norm1 = nn.LayerNorm(D) # 自注意力残差相加后的归一化
        self.norm2 = nn.LayerNorm(D) # FNN 残差相加后的归一化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x 已经过 Embedding 和位置编码，形状为 [S, D]
        # 1-2. 多头自注意力 + 残差 + 归一化
        # Q、K、V 都是 x，所有英文位置可以相互读取
        x = self.norm1(x + self.self_mha(x, x, x))

        # 3-4. 前馈网络 + 残差 + 归一化; 残差使用进入 FNN 前的 x
        x = self.norm2(x + self.ffn(x))
        return x # 输出形状仍为 [S, D], 作为下一层的输入

class Encoder(nn.Module):
    def __init__(self, num_layers) -> None:
        super().__init__()
        # 各层结构相同，参数独立
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 英文输入表示 -> 第一层 -> 第二层 -> ... -> 最后一层输出表示
        for layer in self.layers:
            x = layer(x) # 每层均执行：自注意力 -> 残差 -> 归一化 -> FNN -> 残差 -> 归一化
        return x # 输出形状仍为 [S, D], 作为 Decoder 的输入

class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention() # 多头自注意力
        self.cross_attn = MultiHeadAttention() # 多头交叉注意力
        self.ffn = FeedForward() # FNN: 逐位置加工特征
        self.norm1 = nn.LayerNorm(D) # 自注意力残差相加后的
        self.norm2 = nn.LayerNorm(D) # 交叉注意力残差相加后的归一化
        self.norm3 = nn.LayerNorm(D) # FNN 残差相加后的归一化

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        # x: [T, D], 中文前缀表示；encoder_output: [S, D], 英文编码表示
        # 1-2. 因果多头自注意力 -> Add & Norm; 位置 i 只能读取 0 到 i 的前缀信息
        x = self.norm1(x + self.self_attn(x, x, x, causal=True))
        # 3-4. 多头交叉注意力 -> Add & Norm; 位置 i 可以读取英文编码的所有信息
        # Q 来自中文 x，K、V 来自英文编码表示，输出与 x 形状都是 [T, D]
        x = self.norm2(x + self.cross_attn(x, encoder_output, encoder_output))
        # 5-6. FFN -> Add & Norm; 残差使用进入 FNN 前的 x
        x = self.norm3(x + self.ffn(x))
        return x # 输出形状仍为 [T, D], 作为下一层的

class Decoder(nn.Module):
    def __init__(self, num_layers) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        # 中文前缀表示 -> 第一层 -> 第二层 -> ... -> 最后一层输出表示
        for layer in self.layers:
            # 每层更新中文 x, 并读取同一份英文编码表示 encoder_output，encoder_output 不在此处更新
            x = layer(x, encoder_output)
        return x # 输出形状仍为 [T, D], 后续在投影到中文此表中，计算各 Token 的分数

class MiniTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 两种语言分别使用自己的词向量表
        self.source_embedding = TokenEmbedding(len(source_words)) # 英文词表大小
        self.target_embedding = TokenEmbedding(len(target_words)) # 中文词表大小

        # Encoder 编码英文； Decoder 根据中文前缀读取英文编码结果
        self.encoder = Encoder(NUM_LAYERS)
        self.decoder = Decoder(NUM_LAYERS)

        # 将每个中文位置的 D 维向量表示映射为中文词表中各 Token 的分数，输出形状为 [T, len(target_words)]
        self.output_projection = nn.Linear(D, len(target_words)) # 输出投影到中文词表大小，得到每个位置的各 Token 分数

    def encode(self, source):
        # source 是英文 Token ID 形状为 [S]
        # TokenEmbedding 加入词向量与位置信息：[S] -> [S, D]
        # Encoder 输出仍为 [S, D]
        return self.encoder(self.source_embedding(source))

    def decode(self, target, encoder_output):
        # target 是中文前缀 Token ID: [T]; encoder_output 是英文表示 [S, D]
        x = self.target_embedding(target) # 查词向量、缩放并加位置编码: [T] -> [T, D]
        x = self.decoder(x, encoder_output) # 因果自注意力与交叉注意力处理后仍未[T, D]
        # 输出 logits: [T, 中文词表大小]，每行用于预测对应位置的下一个 Token
        # 训练时直接交给 CrossEntropyLoss, 无需在这里先做 Softmax
        return self.output_projection(x)

    def forward(self, source, target):
        # 完整前向过程：先编码英文，再结合中文前缀计算词表分数
        # 训练时 target 传入完整中文目标去掉末尾 EOS 后的序列
        encoder_output = self.encode(source)
        return self.decode(target, encoder_output)

