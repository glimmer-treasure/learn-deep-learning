import torch
from torch import nn
import random
import matplotlib.pyplot as plt
from .models import MiniTransformer
from .dataset import examples

# 1. 创建模型实例，并固定初始化，方便重复运行时比较结果
# 固定随机种子以确保结果可复现
random.seed(42)
torch.manual_seed(42)
model = MiniTransformer()

# 2. 配置损失函数和优化器
# CrossEntropyloss 接收原始 logits, 不需要提前做 Softmax
criterion = nn.CrossEntropyLoss()
# model.parameters() 收集整个模型的可训练参数；Adam 用学习率 0.003 更新他们
optimizer = torch.optim.Adam(model.parameters(), lr = 0.003)
losses = [] # 记录每轮训练的平均句子损失，用于画图

model.train() # 切换到训练模式；真正的计算和参数更新在下面的循环中进行
for epoch in range(100): # 一轮会遍历全部 8 对句子，共训练 100 轮
    random.shuffle(examples) # 每轮打乱句对顺序，避免总按固定顺序学习
    total_loss = 0.0
    for source, target in examples: # 一次处理一对句子，并更新一次参数
        # 3. 前向计算：英文作为 Encoder 输入，真实中文前缀作为 Decoder 输入
        # target[:-1] 去掉末尾 EOS; 输出 logits 的形状为 [T, 中文词表大小]
        logits = model(source, target[:-1])
        # target[1:] 去掉开头 BOS, 作为各位置要预测的下一个 Token, 形状为 [T]
        # 默认对当前句子中所有预测位置的损失取平均
        loss = criterion(logits, target[1:])

        # 4. 更新参数：先清除上一次的梯度，再计算本次梯度，最后由 Adam 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() # 只累计损失数值，不保留计算图
    
    # 5. 对本轮 8 个句子的损失取平均，记录并定期打印训练进度
    losses.append(total_loss / len(examples))
    print(f"Epoch {epoch + 1:3d} | loss={losses[-1]:.4f}")
        

# 6. 绘制损失曲线，观察模型学习这些句子时的损失是否逐渐下降
plt.plot(range(1, 101), losses)
plt.xlabel("Epoch")
plt.ylabel("Mean sentence loss")
plt.show()
