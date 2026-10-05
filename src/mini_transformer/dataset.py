import torch

en_cn_pairs = [
    ("there is a cat under the table", "桌子 下面 有 一只 猫"),
    ("there is a dog on the chair", "椅子 上面 有 一只 狗"),
    ("there is a cat on the chair", "椅子 上面 有 一只 猫"),
    ("there is a dog under the table", "桌子 下面 有 一只 狗"),
    ("there is a small cat on the table", "桌子 上面 有 一只 小 猫"),
    ("there is a small dog under the chair", "椅子 下面 有 一只 小 狗"),
    ("there is a black cat under the chair", "椅子 下面 有 一只 黑色的 猫"),
    ("there is a black dog on the table", "桌子 上面 有 一只 黑色的 狗"),
]

# 1. 分别收集英文和中文的 token，去重、排序、建立词表
source_words = sorted({word for english, chinese in en_cn_pairs for word in english.split()})
target_words = ["<BOS>", "<EOS>"] + sorted({word for english, chinese in en_cn_pairs for word in chinese.split()})

# 列表用于“ID -> Token”, 字典用于“Token -> ID”
source_ids = { word: i for i, word in enumerate(source_words) }
target_ids = { word: i for i, word in enumerate(target_words) }
BOS = target_ids["<BOS>"]
EOS = target_ids["<EOS>"]

# 2. 将每对句子转换为一维整数张量，中文目标两端加上 BOS 和 EOS
examples = []
for english, chinese in en_cn_pairs:
    source = [source_ids[word] for word in english.split()]
    target = [BOS] + [target_ids[word] for word in chinese.split()] + [EOS]
    examples.append((
        torch.tensor(source, dtype=torch.long),
        torch.tensor(target, dtype=torch.long)
    ))
