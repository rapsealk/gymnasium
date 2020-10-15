#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import argparse
import datetime
from collections import deque
from itertools import count

import gym
import numpy as np

from models.tensorflow_impl import DQN as TensorFlowDQN
# from models.pytorch_impl import DQN as PyTorchDQN


EPSILON_MAX = 1.0
EPSILON_MIN = 0.1
EPSILON_STEP = 1e6
BATCH_SIZE = 32

REPLAY_MEMORY = deque(maxlen=1_000_000)

parser = argparse.ArgumentParser()
parser.add_argument('--tensorflow', action='store_true', default=False)
parser.add_argument('--torch', action='store_true', default=False)
args = parser.parse_args()


if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    model = TensorFlowDQN(env.action_space.n)

    observation = env.reset()
    observation, _, _, _ = env.step(0)
    observation = model.preprocess(observation)
    observation = np.stack([observation, observation, observation, observation], axis=2)

    returns = 0.0

    for episode in count(1):
        for k in range(BATCH_SIZE):
            epsilon = max(EPSILON_MAX - (EPSILON_MAX - EPSILON_MIN) / EPSILON_STEP, EPSILON_MIN)
            if np.random.uniform() < epsilon:
                action = np.random.randint(0, env.action_space.n)
            else:
                action = model.get_pi(observation[np.newaxis, :])
                action = np.squeeze(action)
                action = np.argmax(action[0, 0])
            next_observation, reward, done, info = env.step(action)

            next_observation = model.preprocess(next_observation)
            next_observation = np.append(observation[:, :, 1:], next_observation[:, :, np.newaxis], axis=2)
            returns += reward

            REPLAY_MEMORY.append((observation, action, reward, next_observation, int(not done)))
            observation = next_observation

            env.render()

            done = (info['ale.lives'] == 4)

            if done:
                print('[%s] Episode %3d: %d' % (datetime.datetime.now().isoformat(), episode, returns))

                observation = env.reset()
                observation, _, _, _ = env.step(0)
                observation = model.preprocess(observation)
                observation = np.stack([observation, observation, observation, observation], axis=2)

                returns = 0.0

        batch = np.random.choice([i for i in range(len(REPLAY_MEMORY))], BATCH_SIZE)
        batch = [REPLAY_MEMORY[i] for i in batch]
        loss = model.train(batch)
        print('[%s] Loss: %f' % (datetime.datetime.now().isoformat(), loss))

    env.close()
