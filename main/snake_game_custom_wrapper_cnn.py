import math

import gym
import numpy as np

from snake_game import SnakeGame


## gym(https://github.com/openai/gym)是OPENAI开发一个开源Python库，用于开发和对比强化学习算法的工具箱，兼容大部分数值计算的库，比如 TensorFlow 和 Theano。提供了一系列测试环境——environments，方便我们测试，并且它们有共享的数据接口，以便我们部署通用的算法。
# 继承了gym.Env对象
# gym.Env类定义在https://github.com/openai/gym/blob/master/gym/core.py
# Gym库的Env类有以下常用方法：
    # reset(): 重置环境为初始状态，并返回初始状态的观测值。
    # step(action): 在环境中采取一个动作，并返回下一个状态、奖励、是否完成和其他信息。
    # render(mode='human'): 将环境渲染为可视化的形式，可以选择不同的渲染模式（human、rgb_array等）。
    # close(): 关闭环境，释放相关资源。
    # seed(seed=None): 设置随机数生成器的种子，以便复现实验结果。
    # action_space: 返回动作空间的描述，包括动作的取值范围、类型等。
    # observation_space: 返回观测空间的描述，包括观测值的类型、形状等。
    # reward_range: 返回奖励的取值范围。
    # metadata: 返回环境的元数据，如名称、版本号等。
    # unwrapped: 如果环境是嵌套的，返回最底层的未封装环境。
class SnakeEnv(gym.Env):
    def __init__(self, seed=0, board_size=12, silent_mode=True, limit_step=True):
        super().__init__()
        self.game = SnakeGame(seed=seed, board_size=board_size, silent_mode=silent_mode)
        self.game.reset()

        self.silent_mode = silent_mode

        # 这个函数是在定义一个强化学习环境对象时用于设置动作空间的代码。在这里，动作空间被定义为一个离散空间，其中有 4 个离散的动作。
        # 这个动作空间是通过调用 gym.spaces.Discrete(4) 方法来创建的，它表示一个具有 4 个离散动作的空间。
        # 在强化学习环境中，动作空间表示智能体可以采取的所有可能动作的集合。在这个特定的环境中，智能体只能从 4 个可能的动作中进行选择。
        # 这个动作空间将被传递给强化学习算法，智能体将根据学习到的策略在这个动作空间中选择动作，以最大化预期的总体回报。
        self.action_space = gym.spaces.Discrete(4) # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 3),
            dtype=np.uint8
        )

        # 一条边有12个格子
        self.board_size = board_size
        # 总共是144个格子
        self.grid_size = board_size ** 2 # Max length of snake is board_size^2
        # 初始化蛇的长度
        self.init_snake_size = len(self.game.snake)
        # 蛇最长能增加多长，等于144 - 蛇的初始长度
        self.max_growth = self.grid_size - self.init_snake_size

        self.done = False

        # limit_step默认是true，这里给训练设置了吃一个球的一个步数上限，目的是为了防止学出来的步数太多？
        if limit_step:
            self.step_limit = self.grid_size * 4 # More than enough steps to get the food.
        else:
            self.step_limit = 1e9 # Basically no limit.
        self.reward_step_counter = 0

    def reset(self):
        # 重置游戏
        self.game.reset()

        self.done = False
        self.reward_step_counter = 0

        obs = self._generate_observation()
        return obs
    
    def step(self, action):
        # 由训练过程驱动蛇走一步，得到走这一步之后游戏的情况
        self.done, info = self.game.step(action) # info = {"snake_size": int, "snake_head_pos": np.array, "prev_snake_head_pos": np.array, "food_pos": np.array, "food_obtained": bool}
        obs = self._generate_observation()

        # 奖励分数初始化为0
        reward = 0.0
        # 获得奖励的步数+1
        self.reward_step_counter += 1

        # 如果蛇已经最长了，则游戏成功结束
        if info["snake_size"] == self.grid_size: # Snake fills up the entire board. Game over.
            # 成功奖励=蛇增加的总长度 * 0.1
            reward = self.max_growth * 0.1 # Victory reward
            self.done = True
            if not self.silent_mode:
                self.game.sound_victory.play()
            return obs, reward, self.done, info
        
        # 如果步数超过了吃一个球预设的最大步数，游戏结束
        if self.reward_step_counter > self.step_limit: # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True
        
        # 如果游戏结束
        if self.done: # Snake bumps into wall or itself. Episode is over.
            # Game Over penalty is based on snake size.
            # 这时候奖励是个负数，也就是惩罚，作者在视频中有讲为什么要给惩罚，因为如果不惩罚，可能会死循环，比如一直转圈，即不死也不吃球，吃球就会变长，变长就容易死
            reward = - math.pow(self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth) # (-max_growth, -1)            
            reward = reward * 0.1
            return obs, reward, self.done, info

        # 游戏还没结束，这时候吃到球了  
        elif info["food_obtained"]: # Food eaten. Reward boost on snake size.
            # 现在蛇的长度越大，吃到一个球的奖励越大，最大接近1
            reward = info["snake_size"] / self.grid_size
            ## 吃一个球的步数限制清零
            self.reward_step_counter = 0 # Reset reward step counter
        
        # 没吃到球，也得给算个奖励，告诉这一步的好坏，那怎么来评估这一步的好坏呢？
        else:
            # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not.
            # Not competing with game over penalty or the food eaten reward.
            # np.linalg.norm是NumPy库中的一个函数，用于计算向量或矩阵的范数（norm）。范数是一个度量向量或矩阵的大小的方法。常见的范数有L1范数、L2范数和无穷范数等。
            # 大概意思应该是说如果蛇和食物之间更近了，那么应该有奖励，否则有惩罚
            # 蛇越长降低越小？
            if np.linalg.norm(info["snake_head_pos"] - info["food_pos"]) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"]
            else:
                reward = - 1 / info["snake_size"]
            reward = reward * 0.1

        # max_score: 72 + 14.1 = 86.1
        # min_score: -14.1

        return obs, reward, self.done, info
    
    def render(self):
        self.game.render()

    # 这个函数是一个用于获取动作掩码的方法。它返回一个动作掩码数组，数组的形状是 (1, num_actions)，其中 num_actions 是动作空间的大小。
    # 在这个方法中，它通过遍历动作空间的每一个动作，调用 _check_action_validity(a) 方法来检查每个动作的有效性（是否合法）。
    # 返回的动作掩码数组中，对应的位置为 1 表示该动作是有效的，为 0 表示该动作是无效的。
    # 通过使用动作掩码，可以在某些环境中限制智能体的行动能力，例如在禁止某些动作的情况下，将对应的动作掩码设置为 0，从而让智能体只能选择有效的动作进行执行。
    def get_action_mask(self):
        return np.array([[self._check_action_validity(a) for a in range(self.action_space.n)]])
    
    # Check if the action is against the current direction of the snake or is ending the game.
    # 检查蛇的每一步能走上下左右哪个方向，不能掉头，不能撞墙，不能撞到自己
    def _check_action_validity(self, action):
        current_direction = self.game.direction
        snake_list = self.game.snake
        row, col = snake_list[0]
        if action == 0: # UP
            if current_direction == "DOWN":
                return False
            else:
                row -= 1

        elif action == 1: # LEFT
            if current_direction == "RIGHT":
                return False
            else:
                col -= 1

        elif action == 2: # RIGHT 
            if current_direction == "LEFT":
                return False
            else:
                col += 1     
        
        elif action == 3: # DOWN 
            if current_direction == "UP":
                return False
            else:
                row += 1

        # Check if snake collided with itself or the wall. Note that the tail of the snake would be poped if the snake did not eat food in the current step.
        if (row, col) == self.game.food:
            game_over = (
                (row, col) in snake_list # The snake won't pop the last cell if it ate food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )
        else:
            game_over = (
                (row, col) in snake_list[:-1] # The snake will pop the last cell if it did not eat food.
                or row < 0
                or row >= self.board_size
                or col < 0
                or col >= self.board_size
            )

        if game_over:
            return False
        else:
            return True

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self):
        # 创建一个12 * 12，值全是0的二位数组
        obs = np.zeros((self.game.board_size, self.game.board_size), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        # transpose是NumPy库中的一个函数，用于对指定数组进行转置操作
        # tuple将转置后的数组转换为元组。这是因为np.transpose()返回的是一个数组，而在下一步中需要将其作为索引使用，需要将其转换为元组形式。
        # linspace用于在指定范围内生成等差序列， 200：序列的起始值，50：序列的结束值，len(self.game.snake)：序列的长度，即生成的元素个数。
        # 使用转置后的数组作为索引，将其对应位置的元素替换为生成的等差序列
        obs[tuple(np.transpose(self.game.snake))] = np.linspace(200, 50, len(self.game.snake), dtype=np.uint8)
        
        # Stack single layer into 3-channel-image.
        # 函数np.stack()，用于在指定轴上堆叠多个数组
        # 将名为obs的数组在最后一个轴上进行堆叠。obs数组的形状是 (self.game.board_size, self.game.board_size, 3)，其中最后一个维度的大小为3。
        obs = np.stack((obs, obs, obs), axis=-1)
        
        # Set the snake head to green and the tail to blue
        obs[tuple(self.game.snake[0])] = [0, 255, 0]
        obs[tuple(self.game.snake[-1])] = [255, 0, 0]

        # Set the food to red
        obs[self.game.food] = [0, 0, 255]

        # Enlarge the observation to 84x84
        # 函数np.repeat()，用于在指定轴上对数组进行重复。
        obs = np.repeat(np.repeat(obs, 7, axis=0), 7, axis=1)

        return obs

# Test the environment using random actions
# NUM_EPISODES = 100
# RENDER_DELAY = 0.001
# from matplotlib import pyplot as plt

# if __name__ == "__main__":
#     env = SnakeEnv(silent_mode=False)
    
    # # Test Init Efficiency
    # print(MODEL_PATH_S)
    # print(MODEL_PATH_L)
    # num_success = 0
    # for i in range(NUM_EPISODES):
    #     num_success += env.reset()
    # print(f"Success rate: {num_success/NUM_EPISODES}")

    # sum_reward = 0

    # # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
    # action_list = [1, 1, 1, 0, 0, 0, 2, 2, 2, 3, 3, 3]
    
    # for _ in range(NUM_EPISODES):
    #     obs = env.reset()
    #     done = False
    #     i = 0
    #     while not done:
    #         plt.imshow(obs, interpolation='nearest')
    #         plt.show()
    #         action = env.action_space.sample()
    #         # action = action_list[i]
    #         i = (i + 1) % len(action_list)
    #         obs, reward, done, info = env.step(action)
    #         sum_reward += reward
    #         if np.absolute(reward) > 0.001:
    #             print(reward)
    #         env.render()
            
    #         time.sleep(RENDER_DELAY)
    #     # print(info["snake_length"])
    #     # print(info["food_pos"])
    #     # print(obs)
    #     print("sum_reward: %f" % sum_reward)
    #     print("episode done")
    #     # time.sleep(100)
    
    # env.close()
    # print("Average episode reward for random strategy: {}".format(sum_reward/NUM_EPISODES))
