import time
import random

import torch
from sb3_contrib import MaskablePPO

from snake_game_custom_wrapper_cnn import SnakeEnv

# 判断是否是Apple Silicon GPU
if torch.backends.mps.is_available():
    MODEL_PATH = r"trained_models_cnn_mps/ppo_snake_final"
else:
    MODEL_PATH = r"trained_models_cnn/ppo_snake_final"

NUM_EPISODE = 10

RENDER = True
FRAME_DELAY = 0.05 # 0.01 fast, 0.05 slow
ROUND_DELAY = 5

seed = random.randint(0, 1e9)
print(f"Using seed = {seed} for testing.")

if RENDER:
    env = SnakeEnv(seed=seed, limit_step=False, silent_mode=False)
else:
    env = SnakeEnv(seed=seed, limit_step=False, silent_mode=True)

# Load the trained model
model = MaskablePPO.load(MODEL_PATH)

total_reward = 0
total_score = 0
min_score = 1e9
max_score = 0

# 玩10次游戏
for episode in range(NUM_EPISODE):
    obs = env.reset()
    episode_reward = 0
    done = False
    
    num_step = 0
    info = None

    sum_step_reward = 0

    retry_limit = 9
    print(f"=================== Episode {episode + 1} ==================")
    while not done:
        # MaskablePPO的predict函数通常具有以下几个参数：
            # observation：用于预测动作的观测值或状态。
            # state：可选参数，用于传递模型的内部状态信息。在序列预测或循环神经网络模型中可能会使用到。
            # mask：可选参数，代表动作的掩码。它可以指定哪些动作是有效的，哪些动作是被掩盖或禁用的。用于实现对动作空间进行限制或约束的功能。
            # deterministic：可选参数，用于控制是否使用确定性策略。如果设置为True，则选择具有最高概率的动作；如果设置为False，则根据概率分布随机选择动作。
            # use_gae：可选参数，表示是否使用GAE（Generalized Advantage Estimation）的方法来估计优势值函数。
            # **kwargs：其他可选的参数，根据具体的实现可能会有所不同，例如温度参数、熵系数等。
        action, _ = model.predict(obs, action_masks=env.get_action_mask())
        prev_mask = env.get_action_mask()
        prev_direction = env.game.direction
        num_step += 1
        obs, reward, done, info = env.step(action)

        if done:
            if info["snake_size"] == env.game.grid_size:
                print(f"You are BREATHTAKING! Victory reward: {reward:.4f}.")
            else:
                last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
                print(f"Gameover Penalty: {reward:.4f}. Last action: {last_action}")

        elif info["food_obtained"]:
            print(f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}")
            sum_step_reward = 0 

        else:
            sum_step_reward += reward
            
        episode_reward += reward
        if RENDER:
            env.render()
            time.sleep(FRAME_DELAY)

    episode_score = env.game.score
    if episode_score < min_score:
        min_score = episode_score
    if episode_score > max_score:
        max_score = episode_score
    
    snake_size = info["snake_size"] + 1
    print(f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, Total Steps: {num_step}, Snake Size: {snake_size}")
    total_reward += episode_reward
    total_score += env.game.score
    if RENDER:
        time.sleep(ROUND_DELAY)

env.close()
print(f"=================== Summary ==================")
print(f"Average Score: {total_score / NUM_EPISODE}, Min Score: {min_score}, Max Score: {max_score}, Average reward: {total_reward / NUM_EPISODE}")
