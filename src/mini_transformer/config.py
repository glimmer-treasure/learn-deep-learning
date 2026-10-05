# 模型参数
D = 32 # 每个 token 的向量维度
HEADS = 2 # 注意力头的数量；D 必须能被 HEADS 整除
D_FF = 64 # 前馈网络的隐藏层维度
NUM_LAYERS = 2 # Encoder 和 Decoder 各自堆叠的层数
MAX_LENGTH = 32 # 位置编码表支持的最大 Token 数