#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import gym

from util.animator import display_frames_as_gif

ENV = 'Breakout-v0'


if __name__ == "__main__":
    env = gym.make(ENV)

    print(env.observation_space)

    print(env.action_space)
    print(env.unwrapped.get_action_meanings())

    observation = env.reset()
    # plt.imshow(observation)
    # plt.show()

    frames = []

    for step in range(1000):
        env.render()
        frames.append(observation)
        observation, reward, done, info = env.step(env.action_space.sample())
        if done:
            break

    display_frames_as_gif(frames, 'breakout.gif')
