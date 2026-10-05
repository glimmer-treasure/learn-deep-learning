import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)  # For reproducibility

plt.rcParams['font.sans-serif'] = ['Heiti TC', 'PingFang SC', 'Arial Unicode MS']  # Set font to support Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Display negative signs correctly

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

# print(f'训练集：{len(train_set)} 张')
# print(f'测试集：{len(test_set)} 张')
# print(f'每张图：{train_set[0][0].shape} (通道 × 高 × 宽)')

# fig, axes = plt.subplots(3, 3, figsize=(6, 6))
# for i, ax in enumerate(axes.flat):
#     image, label = train_set[i]
#     ax.imshow(image.squeeze() * 0.3081 + 0.1307, cmap='gray')  # Unnormalize the image
#     ax.set_title(f'标签: {label}', fontsize=12)
#     ax.axis('off')
# plt.suptitle('MNIST 训练集前9张', fontsize=13, y=1.02)
# plt.tight_layout()
# plt.show()

torch.manual_seed(42)  # For reproducibility

W1 = (torch.randn(784, 128) * (2.0 / 784) ** 0.5).requires_grad_()
b1 = torch.zeros(128, requires_grad=True)

W2 = (torch.randn(128, 10) * (2.0 / 128) ** 0.5).requires_grad_()
b2 = torch.zeros(10, requires_grad=True)

params = [W1, b1, W2, b2]
lr = 0.1

# print(f'网络共 {sum(p.numel() for p in params)} 个参数')

train_losses = []
test_accs = []

for epoch in range(5):
    running_loss = 0.0
    n_samples = 0

    for xb, yb in train_loader:
        x = xb.view(-1, 784)  # Flatten the images
        h = F.relu(x @ W1 + b1)  # Hidden layer with ReLU activation
        logits = h @ W2 + b2  # Output layer

        loss = F.cross_entropy(logits, yb)  # Compute loss
        loss.backward()  # Backpropagation

        with torch.no_grad():
            for p in params:
                p -= lr * p.grad  # Update parameters
                p.grad.zero_()  # Zero the gradients  

        running_loss += loss.item() * yb.size(0)
        n_samples += yb.size(0)
    train_loss = running_loss / n_samples

    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            x = xb.view(-1, 784)
            logits = F.relu(x @ W1 + b1) @ W2 + b2
            correct += (logits.argmax(1) == yb).sum().item()
    test_acc = correct / len(test_set)

    train_losses.append(train_loss)
    test_accs.append(test_acc)
    print(f'epoch {epoch + 1} / 5 train_loss = {train_loss:.4f}, test_acc = {test_acc:.4f}')

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the images
        x = F.relu(self.fc1(x))
        return self.fc2(x)

torch.manual_seed(42)  # For reproducibility
model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

train_losses_v2 = []
test_accs_v2 = []

for epoch in range(5):
    model.train()
    running_loss = 0.0
    n_samples = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * yb.size(0)
        n_samples += yb.size(0)

    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            correct += (model(xb).argmax(1) == yb).sum().item()
    train_losses_v2.append(running_loss / n_samples)
    test_accs_v2.append(correct / len(test_set))
    print(f'epoch {epoch + 1} / 5 train_loss = {train_losses_v2[-1]:.4f}, test_acc = {test_accs_v2[-1]:.4f}')