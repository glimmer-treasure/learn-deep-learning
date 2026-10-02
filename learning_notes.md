# Deep Learning Study Notes & FAQ (深度学习学习笔记与问答知识库)

本总结文档用于记录在深度学习、Transformer 架构手写实现以及环境配置过程中遇到的各种技术问题与解决方案，方便后续复习、迭代与扩充。

---

## ⚙️ 第一部分：Python 环境与包管理工具 (uv)

### Q1：`uv venv` 命令的作用是什么？
* **A：** 用于快速创建 Python 虚拟环境。它基于 Rust 编写，速度比标准 `python -m venv` 快 10 到 100 倍。若本地缺失目标 Python 版本，它还具备**自动下载并安装**对应 Python 解释器的功能。

### Q2：在 `uv` 项目中不手动创建 `.venv`，依赖还会安装在虚拟环境中吗？
* **A：** **是的，只要你使用的是现代 `uv` 项目命令**。当在包含 `pyproject.toml` 的项目目录下执行 `uv run`、`uv add` 或 `uv sync` 时，如果 `uv` 发现当前没有 `.venv` 文件夹，它会在后台**自动、强制创建虚拟环境**并隔离安装依赖，绝对不会污染你的系统全局环境。

### Q3：为什么找不到 `uv install` 命令？该用什么替代？
* **A：** `uv` 的原生命令体系中**没有**独立的 `uv install`。它将“安装”功能拆分为了更精准的子命令：
  * **老项目/传统模式（只有 requirements.txt）**：使用 `uv pip install -r requirements.txt`。
  * **现代化新项目（有 pyproject.toml）**：使用 `uv add 包名`（添加单个包）或 `uv sync`（一键对齐锁文件安装所有依赖）。
  * **全局独立工具安装**：使用 `uv tool install 包名`（如安装全局可用的 `ruff` 等工具，会自动建立全局隔离环境）。

---

## 🖼️ 第二部分：深度学习与 Apple 芯片环境配置

### Q4：使用 `uv` 安装 PyTorch 时，`torch`、`torchvision` 和 `torchaudio` 都是必须的吗？
* **A：** **不是，只有 `torch` 是完全核心且必须的**。`torchvision`（计算机视觉）和 `torchaudio`（音频处理）是独立的官方扩展库。对于从零手写 mini-Transformer 这样的纯文本/架构学习项目，**只需要执行 `uv add torch` 即可**。不安装其他两个可以极大节省数 GB 的磁盘空间，还能减少 `uv.lock` 在版本解析时的潜在依赖冲突。

### Q5：在 Apple M2 Max 芯片上，如何让 PyTorch 调用 GPU 硬件加速？
* **A：** 苹果 M 系列芯片使用的是 **MPS (Metal Performance Shaders)** 后端来实现硬件加速。通过 `uv add torch` 安装的标准版已原生内置支持。在编写代码时，需要通过以下代码将模型和张量搬运到 GPU 上计算：
  ```python
  import torch
  device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
  # 示例：model = MiniTransformer().to(device)
  ```

---

## 🔧 第三部分：版本控制契约与冲突解决

### Q6：执行命令时报 `error: The Python request from .python-version ... is incompatible with the project's Python requirement...` 是怎么回事？
* **A：** 这是因为你本地通过 `.python-version` 文件指定的本地开发 Python 版本（例如 `3.12.15`），**低于**项目 `pyproject.toml` 中 `requires-python` 字段规定的硬性下限门槛（例如 `>=3.14`）。`uv` 为了防止由于 Python 版本过低引发的语法或运行崩溃，直接在依赖安装阶段进行了拦截报错。

### Q7：`.python-version` 文件的优先级不是最高吗？为什么还会发生冲突？
* **A：** 本地优先级最高是指“如果你选了一个符合项目契约范围的版本，它会覆盖一切默认设置”。但它的前提是**本地选择的版本不能违法（不能超出项目声明的兼容底线）**。当 `pyproject.toml` 规定必须 `>=3.14` 时，即使 `.python-version` 拥有最高本地选择权，也不能强行指定不兼容的 `3.12`。

### Q8：如果只改 `pyproject.toml` 的版本要求，而不运行 `uv python pin` 会怎么样？
* **A：** 如果不 `pin` 且删除了 `.python-version`，`uv` 在检测到 `requires-python = ">=3.12"` 时，会默认去寻找或下载你电脑上/网络上**满足条件的最高版本（如最新的 3.13 或 3.14）**。由于 PyTorch 等深度学习库对极高版本 Python 生态支持尚未完全稳定，让 `uv` 自由升级极大概率导致后续安装报错或编译失败。

### Q9：`uv python pin 3.12` 命令的具体作用是什么？
* **A：** 该命令会在本地生成或更新 `.python-version` 文件，明确告诉 `uv` ：“这个项目在本地开发时，**死死锁定在 Python 3.12**”。此后无论是创建环境、安装依赖还是执行 `uv run`，都会强制自动采用 Python 3.12，完美避开高版本 Python 的不兼容地雷，是确保深度学习项目环境稳定的最佳实践。

---

## 📈 后续扩充指南（你可以这样继续记录...）
1. **[在此处添加问题]**：简述你在后续手写 Transformer 块、注意力机制或训练循环中遇到的 Bug。
2. **[在此处添加答案]**：记录导致问题的根本原因以及 M2 Max 上的具体表现和修复代码。