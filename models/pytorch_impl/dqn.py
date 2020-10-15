#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, n):
        super(DQN, self).__init__()
        self.hidden1 = nn.Conv2d(4, 16, kernel_size=(8, 8), stride=4)
        self.hidden2 = nn.Conv2d(16, 32, kernel_size=(4, 4), stride=2)
        self.hidden3 = nn.Linear(32, 256)
        self.hidden4 = nn.Linear(256, n)

        self.optimizer = optim.RMSprop(lr=1e-3)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        return F.softmax(self.hidden4(x))

    def preprocess(self, observation):
        assert observation.shape == (210, 160, 3)
        observation = tf.image.rgb_to_grayscale(observation)
        observation = tf.image.resize(observation, (110, 84))
        # tf.image.crop_and_resize()

        observation = np.asarray(observation)
        observation = np.squeeze(observation)
        observation = observation[-84:, :]

        assert observation.shape == (84, 84)

        return observation.astype(np.float32)


if __name__ == "__main__":
    pass
