# SnakeAI

简体中文 | [English](README.md) | [日本語](README_JA.md)

本项目包含经典游戏《贪吃蛇》的程序脚本以及可以自动进行游戏的人工智能代理。该智能代理基于深度强化学习进行训练，包括两个版本：基于多层感知机（Multi-Layer Perceptron）的代理和基于卷积神经网络（Convolution Neural Network）的代理，其中后者的平均游戏分数更高。

### 文件结构

```bash
├───main
│   ├───logs
│   ├───trained_models_cnn
│   ├───trained_models_mlp
│   └───scripts
├───utils
│   └───scripts
```

项目的主要代码文件夹为 `main/`。其中，`logs/` 包含训练过程的终端文本和数据曲线（使用 Tensorboard 查看）；`trained_models_cnn/` 与 `trained_models_mlp/` 分别包含卷积网络与感知机两种模型在不同阶段的模型权重文件，用于在 `test_cnn.py` 与 `test_mlp.py` 中运行测试，观看两种智能代理在不同训练阶段的实际游戏效果。

另一个文件夹 `utils/` 包括两个工具脚本。`check_gpu_status/` 用于检查 GPU 是否可以被 PyTorch 调用；`compress_code.py` 可以将代码缩进、换行全部删去变成一行紧密排列的文本，方便与 GPT-4 进行交流，向 AI 询问代码建议（GPT-4 对代码的理解能力远高于人类，不需要缩进、换行等）。

## 运行指南

本项目基于 Python 编程语言，用到的外部代码库主要包括 [Pygame](https://www.pygame.org/news)、[OpenAI Gym](https://github.com/openai/gym)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) 等。程序运行使用的 Python 版本为 3.8.16，建议使用 [Anaconda](https://www.anaconda.com) 配置 Python 环境。以下配置过程已在 Windows 11 系统上测试通过。以下为控制台/终端（Console/Terminal/Shell）指令。

### 环境配置

```bash
# 创建 conda 环境，将其命名为 SnakeAI，Python 版本 3.8.16
conda create -n SnakeAI python=3.8.16
conda activate SnakeAI
```

在 Windows 与 macOS 下配置外部代码库的过程略有不同。Windows 下使用 CUDA 加速，macOS 下则使用 MPS (Metal Performance Shaders) 进行加速，且需要降级 `pip` 与 `setuptools`。

Windows:
```bash 
# 使用 GPU 训练需要手动安装完整版 PyTorch
conda install pytorch=2.0.0 torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 运行程序脚本测试 PyTorch 是否能成功调用 GPU
python .\utils\check_gpu_status.py

# 安装外部代码库
pip install -r requirements.txt
```

macOS (Apple Silicon):
```bash
# 使用 GPU 训练需要手动安装 Apple Silicon 版 PyTorch
conda install pytorch::pytorch=2.0.1 torchvision torchaudio -c pytorch

# 运行程序脚本测试 PyTorch 是否能成功调用 GPU
python utils/check_gpu_status_mps.py

# 安装 tensorboard
pip install tensorboard==2.13.0

# 降级安装外部代码库
pip install setuptools==65.5.0 pip==21
pip install -r requirements.txt
```

### 运行测试

项目 `main/` 文件夹下包含经典游戏《贪吃蛇》的程序脚本，基于 [Pygame](https://www.pygame.org/news) 代码库，可以直接运行以下指令进行游戏：

```bash
cd [项目上级文件夹]/snake-ai/main
python .\snake_game.py
```

环境配置完成后，可以在 `main/` 文件夹下运行 `test_cnn.py` 或 `test_mlp.py` 进行测试，观察两种智能代理在不同训练阶段的实际表现。

```bash
cd [项目上级文件夹]/snake-ai/main
python test_cnn.py
python test_mlp.py
```

模型权重文件存储在 `main/trained_models_cnn/` 与 `main/trained_models_mlp/` 文件夹下。两份测试脚本均默认调用训练完成后的模型。如果需要观察不同训练阶段的 AI 表现，可将测试脚本中的 `MODEL_PATH` 变量修改为其它模型的文件路径。

### 训练模型

如果需要重新训练模型，可以在 `main/` 文件夹下运行 `train_cnn.py` 或 `train_mlp.py`。

```bash
cd [项目上级文件夹]/snake-ai/main
python train_cnn.py
python train_mlp.py
```

### 查看曲线

项目中包含了训练过程的 Tensorboard 曲线图，可以使用 Tensorboard 查看其中的详细数据。推荐使用 VSCode 集成的 Tensorboard 插件直接查看，也可以使用传统方法：

```bash
cd [项目上级文件夹]/snake-ai/main
tensorboard --logdir=logs/
```

在浏览器中打开 Tensorboard 服务默认地址 `http://localhost:6006/`，即可查看训练过程的交互式曲线图。

## 鸣谢
本项目调用的外部代码库包括 [Pygame](https://www.pygame.org/news)、[OpenAI Gym](https://github.com/openai/gym)、[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) 等。感谢各位软件工作者对开源社区的无私奉献！

本项目使用的卷积神经网络来自 Nature 论文：

[1] [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

## From ChatGPT：什么是强化学习
* 强化学习（Reinforcement Learning）是机器学习领域的一个分支，它涉及到智能体（Agent）在与环境进行交互的过程中，通过试错学习来获得最大化的累积奖励。强化学习的目标是让智能体基于观察和奖励来学习最优的行为策略。
* 在强化学习中，智能体通过与环境交互来获取环境的状态信息，并根据当前状态选择一个行动。环境根据智能体的行动和当前状态给予一个奖励信号，用以评估智能体的行为好坏。智能体根据奖励信号来调整自己的行为策略，以使得未来的累积奖励最大化。
* 强化学习的关键概念包括：智能体、环境、状态、行动和奖励。智能体通过采取不同的行动来改变环境的状态，并通过观察环境的反馈来学习最优的行为策略。
* 强化学习被广泛应用于自动驾驶、机器人控制、游戏设计等领域，它可以通过与环境的交互来学习复杂的任务和决策问题，具有很强的灵活性和适应性。

## From ChatGPT：一个基于强化学习和卷积神经网络的自动玩贪吃蛇的AI程序的实现原理如下：

* 状态表示：将贪吃蛇游戏的当前状态转化为一个输入向量或图像。可以使用一些特征工程方法来提取有关蛇的位置、食物的位置等信息。
* 策略网络：使用卷积神经网络（CNN）作为策略网络，将状态向量或图像作为输入，输出一个动作的概率分布。可以使用多层卷积层和全连接层来提取和处理状态的特征。
* 强化学习算法：使用强化学习算法（如深度Q网络（DQN）或策略梯度（PG）方法）来训练策略网络。这些算法通过与环境进行交互来优化策略网络的参数，使得AI程序能够学习到最优的策略。
* 奖励设计：定义适当的奖励函数来指导AI程序的学习过程。例如，对于吃到食物给予正奖励，对于撞到墙或自身给予负奖励。
* 经验回放：使用经验回放缓冲区来存储AI程序与环境的交互经验。在训练过程中，从经验回放缓冲区中随机采样一批经验，用于训练策略网络，以减小样本之间的相关性。
* 探索与利用：在训练过程中，使用ε-greedy等策略来平衡探索（尝试新动作）和利用（选择已知最佳动作）之间的权衡。
* 通过不断的训练和优化，AI程序可以学习到贪吃蛇游戏中的最佳策略，使得蛇能够自动地找到食物并避免撞到墙或自身。这种方法将强化学习与卷积神经网络相结合，能够处理高维状态空间和动作空间，并能够从原始输入中学习到有效的表示和策略。
