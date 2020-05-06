#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import gym
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

print(tf.__version__)

env = gym.make("CartPole-v1")
goal_steps = 500


def data_preparation(n, k, f, render=False):
    game_data = []
    for i in range(n):
        score = 0
        game_steps = []
        observation = env.reset()
        for step in range(goal_steps):
            if render:
                env.render()
            action = f(observation)
            game_steps.append((observation, action))
            observation, reward, done, info = env.step(action)
            score += reward
            if done:
                break
        game_data.append((score, game_steps))
    game_data.sort(key=lambda x: -x[0])

    training_set = []
    for i in range(k):
        for step in game_data[i][1]:
            if step[1] == 0:
                training_set.append((step[0], (1, 0)))
            else:
                training_set.append((step[0], (0, 1)))

    print('{0}/{1}th score: {2}'.format(k, n, game_data[k-1][0]))

    if render:
        for i in game_data:
            print('Score: {0}'.format(i[0]))

    return training_set


def build_model():
    model = Sequential([
        Dense(8, input_dim=4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(loss='mse', optimizer=Adam())
    model.summary()
    return model


def train_model(model, training_set):
    X = np.array([i[0] for i in training_set]).reshape(-1, 4)
    y = np.array([i[1] for i in training_set]).reshape(-1, 2)
    model.fit(X, y, epochs=10)


if __name__ == "__main__":
    N = 1000
    K = 50
    model = build_model()

    def predictor(x):
        return np.random.choice([0, 1], p=model.predict(x.reshape(-1, 4))[0])

    while True:
        training_data = data_preparation(N, K, predictor, True)
        train_model(model, training_data)
