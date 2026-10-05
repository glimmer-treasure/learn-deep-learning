import torch
from .train import model
from .dataset import source_ids, target_words, BOS, EOS
from .config import MAX_LENGTH

@torch.no_grad()
def translate(english):
    model.eval()

    # 1. 将英文按空格分词，再查词表得到一维 Token ID 张量，形状为 [S]
    source = torch.tensor([source_ids[word] for word in english.split()])
    # 英文只需编码一次，得到 [S, D] 的表示，后续每一步生成都复用它
    encoder_output = model.encode(source)
    prefix = [BOS] # 中文前缀从 BOS 开始，后续逐步追加模型自己生成的 Token

    # 2. 逐个生成中文 Token, 最多生成 16 步，同时保证输入不超出位置表容量
    for step in range(min(16, MAX_LENGTH)):
        target = torch.tensor(prefix) # 将当前完整前缀转为 ID 张量，形状为[T]
        # Decoder 读取中文前缀和英文表示，输出[T, 中文词表大小] 的分数
        logits = model.decode(target, encoder_output)
        # 只取最后一个位置的分数，用来预测前缀之后的下一个 Token
        # 选择分数最高的 Token ID
        next_id = logits[-1].argmax().item()
        if next_id == EOS:
            break # 预测到结束标记就停止，不将 EOS 加入译文
        prefix.append(next_id) # 将预测结果追加到前缀，作为下一步的输入

    # 3. 去掉开头的 BOS, 将其余 ID 转回中文 Token, 在拼成译文
    return "".join(target_words[i] for i in prefix[1:])

