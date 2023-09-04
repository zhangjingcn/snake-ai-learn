import os
import sys
import random

import torch
# stable_baselines3(https://github.com/DLR-RM/stable-baselines3) 是一个实现了强化学习算法的python库
from stable_baselines3.common.monitor import Monitor
# Monitor 是一个用于监控和记录训练过程的类。它可以包装一个环境对象，并在每个训练步骤结束时保存训练相关的信息，
# 如奖励值、步数等。通过监控器，可以将训练过程中的数据保存到文件中，方便后续分析和可视化。
from stable_baselines3.common.vec_env import SubprocVecEnv
# SubprocVecEnv 是一个用于创建多个并行运行的环境的类。它可以将多个环境对象进行并行化处理，从而提高训练效率。
# 通过将多个环境对象放入子进程中运行，可以同时进行多个环境的交互，从而加速训练过程。
from stable_baselines3.common.callbacks import CheckpointCallback
# CheckpointCallback 是一个用于设置模型检查点的回调函数类。它可以在训练过程中的指定时间间隔内自动保存模型的参数到文件中，
# 以便后续可以恢复训练或进行评估。通过设置检查点回调，可以在训练过程中定期保存模型的参数，防止训练过程中的意外中断导致的训练进度丢失

from sb3_contrib import MaskablePPO
# MaskablePPO 是一个来自于 sb3_contrib 模块的基于 PPO 算法的策略优化类。sb3_contrib 是 Stable Baselines3 的一个扩展模块，
# 包含了一些不在 Stable Baselines3 官方支持列表中的算法和功能。MaskablePPO 支持使用掩码对动作空间进行屏蔽，
# 从而在某些环境中限制智能体的行动选择。

# Q：PPO算法是什么
# A：
## PPO（Proximal Policy Optimization）是一种基于策略优化的强化学习算法，广泛应用于连续动作空间和离散动作空间的强化学习任务中。
## PPO 是基于 Trust Region Policy Optimization（TRPO）算法的改进版本，旨在解决 TRPO 的计算效率问题。
## PPO 通过使用一种称为 Clipped Surrogate Objective 的近似方法，来更新策略网络的参数。这种近似方法在更新策略参数时，引入一个判断指标，
## 用于衡量当前策略更新是否过大，如果过大，则将更新限制在一个可接受的范围内。
## PPO 的核心思想是通过优化一个逼近目标函数来更新策略网络的参数，目标函数是由两个部分组成：
## 一个是策略在当前参数下的性能，另一个是策略在更新参数后的性能。通过最小化目标函数，PPO 可以在保证策略更新不会太大的情况下，寻找到更优的策略参数。
## PPO 算法的特点是相对简单、易于实现，并且在许多强化学习任务中表现出良好的性能和稳定性。
## 它被广泛应用于各种实际问题中，如机器人控制、游戏玩法、交通调度等领域。

from sb3_contrib.common.wrappers import ActionMasker
# ActionMasker 是一个用于动作空间掩码的类。它可以将一个环境对象进行包装，并根据指定的掩码规则对动作空间进行限制。
# 通过设置掩码规则，可以在某些环境中限制智能体的行动能力，从而达到一些特定的训练效果。
# 例如，在贪吃蛇游戏中，可以设置掩码规则，使得蛇不能选择走回头路或者直接碰到自己，从而让智能体更加聚焦在学习正确的策略上。
# sb3_contrib 模块中的 common 子模块中包含了一些常用的环境包装器，例如 Monitor、VecNormalize 等，
# 这些环境包装器可以加速训练，提高训练效果，也可以使训练过程更加稳定和可靠。

# sb3_contrib(https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)单独提供了一些实验性的强化学习算法

from snake_game_custom_wrapper_cnn import SnakeEnv

# 判断是否是Apple Silicon GPU
if torch.backends.mps.is_available():
    NUM_ENV = 32 * 2
else:
    NUM_ENV = 32
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

#这个函数是一个线性学习率调度器，在训练神经网络时可以使用。它接受两个参数：initial_value 和 final_value，分别表示初始学习率和最终学习率。默认最终学习率为 0.0。
#函数先检查 initial_value 是否为字符串类型，如果是，则将其转换为浮点数。这是为了方便用户输入。然后检查 initial_value 是否大于 0.0，如果不是，则抛出异常。
#函数返回一个函数 scheduler，它接受一个参数 progress，表示训练的进度。progress 为 0.0 时，返回最终学习率；为 1.0 时，返回初始学习率。在 0.0 和 1.0 之间的进度值，返回的学习率是根据线性插值计算得到的。最终返回的 scheduler 函数可以被用作优化器的学习率调度器。

# Q：
## 线性学习率调度器在训练神经网络时是如何使用的？
# A：
## 在线性学习率调度器中，初始学习率和最终学习率是预先设置的参数。在训练神经网络过程中，可以通过调用调度器函数来根据当前的训练进度动态地获取学习率。
## 通常，在每个训练步骤中，需要根据当前的训练迭代次数或者训练样本的进度来获取对应的学习率。这个进度值一般是一个从0到1的值，表示训练的进程。
## 在训练过程中，可以在优化器的学习率参数中使用该线性学习率调度器来获取动态变化的学习率。

# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):

    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert (initial_value > 0.0)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler

def make_env(seed=0):
    def _init():
        env = SnakeEnv(seed=seed)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        env.seed(seed)
        return env
    return _init

def main():

    # Generate a list of random seeds for each environment.
    seed_set = set()
    while len(seed_set) < NUM_ENV:
        seed_set.add(random.randint(0, 1e9))

    # Create the Snake environment.
    env = SubprocVecEnv([make_env(seed=s) for s in seed_set])

    if torch.backends.mps.is_available():
        lr_schedule = linear_schedule(5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)
        # Instantiate a PPO agent using MPS (Metal Performance Shaders).
        model = MaskablePPO(
            "CnnPolicy",
            env,
            device="mps",
            verbose=1,
            n_steps=2048,
            batch_size=512*8,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR
        )
    else:
        lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
        clip_range_schedule = linear_schedule(0.150, 0.025)
        # Instantiate a PPO agent using CUDA.
        # 这段代码中，创建了一个 MaskablePPO 对象，并且设置了一些参数。下面是每个参数的解释：
        # "CnnPolicy"：指定了模型的策略网络类型。这里使用的是一个 CnnPolicy，即卷积神经网络策略网络。
        # env：强化学习环境对象，用于指定训练的环境。
        # device="cuda"：指定了模型在训练时使用的设备，这里设置为使用 CUDA 加速计算的 GPU 设备。
        # verbose=1：指定了输出的详细程度，1 表示输出详细的训练信息。
        # n_steps=2048：指定了每个训练迭代步骤中采集的样本数量。
        # batch_size=512：指定了每个训练批次中使用的样本数量。
        # n_epochs=4：指定了每个训练批次的迭代次数。
        # gamma=0.94：指定了折扣因子，用于计算未来奖励的折扣。
        # learning_rate=lr_schedule：指定了学习率的调度器，用于动态调整学习率。
        # clip_range=clip_range_schedule：指定了 PPO 算法中的剪辑范围，用于控制策略更新的幅度。
        # tensorboard_log=LOG_DIR：指定了训练日志保存的目录，用于 TensorBoard 的可视化。
        # 其他可能存在的参数，例如 ent_coef、vf_coef 等没有在代码中列出。
        # 这些参数用于配置 MaskablePPO 模型，指定了模型的架构、训练环境、训练设备、训练参数等，以便进行强化学习的训练过程。
        model = MaskablePPO(
            "CnnPolicy",
            env,
            device="cuda",
            verbose=1,
            n_steps=2048,
            batch_size=512,
            n_epochs=4,
            gamma=0.94,
            learning_rate=lr_schedule,
            clip_range=clip_range_schedule,
            tensorboard_log=LOG_DIR
        )

    # Set the save directory
    if torch.backends.mps.is_available():
        save_dir = "trained_models_cnn_mps"
    else:
        save_dir = "trained_models_cnn"
    os.makedirs(save_dir, exist_ok=True)

    # 每训练50W Step回调一下，保持一下当前的模型
    checkpoint_interval = 15625 # checkpoint_interval * num_envs = total_steps_per_checkpoint
    checkpoint_callback = CheckpointCallback(save_freq=checkpoint_interval, save_path=save_dir, name_prefix="ppo_snake")

    # Writing the training logs from stdout to a file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, 'w') as log_file:
        sys.stdout = log_file

        # MaskablePPO的learn函数通常有以下几个参数：
            #total_timesteps：训练的总步数，即模型将执行的总环境交互次数。
            #callback：在训练过程中调用的回调函数，用于实现自定义的训练过程中的额外操作。
            #log_interval：指定每个log_interval步打印一次训练信息。
            #tb_log_name：用于指定TensorBoard日志目录的名称。
            #reset_num_timesteps：是否在训练之前重置模型的timesteps计数器。
            #mask_fn：自定义的掩码函数，用于计算动作掩码，根据环境的状态和其他信息确定哪些动作是有效的
        model.learn(
            total_timesteps=int(100000000),
            callback=[checkpoint_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    # Save the final model
    model.save(os.path.join(save_dir, "ppo_snake_final.zip"))

if __name__ == "__main__":
    main()
