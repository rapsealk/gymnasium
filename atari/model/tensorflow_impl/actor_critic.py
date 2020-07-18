#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import time

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import animation

from dqn import ProbabilityDistribution

# TensorFlow CUDA GPU configuration
physical_devices = tf.config.experimental.list_physical_devices("GPU")
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

ID = int(time.time() * 1000)


class AdvancedActorCritic(tf.keras.Model):

    def __init__(self, input_shape, n_outputs):
        super(AdvancedActorCritic, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, input_shape=input_shape)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(128)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.lstm = tf.keras.layers.LSTM(128, stateful=True, recurrent_initializer='he_uniform') # return_sequences
        self.action_fc = tf.keras.layers.Dense(64, activation='relu')
        self.value_fc = tf.keras.layers.Dense(32, activation='relu')
        self.action_head = tf.keras.layers.Dense(n_outputs, activation='softmax', name='action')
        self.value_head = tf.keras.layers.Dense(1, name='value')

    def call(self, inputs, training=False):
        x = tf.keras.activations.relu(self.bn1(self.fc1(inputs)))
        x = tf.keras.activations.relu(self.bn2(self.fc2(x)))

        x = tf.keras.layers.Flatten()(x)
        x = tf.expand_dims(x, axis=0)
        x = self.lstm(x)

        logits = self.action_head(self.action_fc(x))
        value = self.value_head(self.value_fc(x))

        return logits, value


"""BattleZone-ram-v0
class AdvancedActorCritic(tf.keras.Model):
    # Best 100-episode average reward was 29210.00 ± 926.87.
    def __init__(self, input_shape, n_actions):
        super(AdvancedActorCritic, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(4, 4), input_shape=input_shape)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D()
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(3, 3))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D()
        #self.conv3 = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2))
        #self.bn3 = tf.keras.layers.BatchNormalization()
        #self.pool3 = tf.keras.layers.MaxPool2D()
        self.lstm = tf.keras.layers.LSTM(16)
        self.action_fc = tf.keras.layers.Dense(8, activation='relu')
        self.value_fc = tf.keras.layers.Dense(8, activation='relu')
        self.action_head = tf.keras.layers.Dense(n_actions, activation='softmax', name='action')
        self.value_head = tf.keras.layers.Dense(1, name='value')

    def call(self, inputs, training=False):
        x = self.pool1(tf.keras.activations.relu(self.bn1(self.conv1(inputs))))
        x = self.pool2(tf.keras.activations.relu(self.bn2(self.conv2(x))))
        #x = self.pool3(tf.keras.activations.relu(self.bn3(self.conv3(x))))

        x = tf.keras.layers.Flatten()(x)
        x = self.lstm(tf.expand_dims(x, axis=0))

        logits = self.action_head(self.action_fc(x))
        value = self.value_head(self.value_fc(x))

        return logits, value
"""


class Agent:

    def __init__(self, env):
        self.env = env

        self.lr = 1e-3
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.01

        self.rollout = 128
        self.batch_size = 128
        self.state_size = env.reset().shape
        self.action_size = env.action_space.n

        self.model = AdvancedActorCritic(self.state_size, self.action_size)
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        self.prob = ProbabilityDistribution()

    def get_action(self, state):
        if np.random.random() > self.epsilon:
            state = tf.convert_to_tensor([state], dtype=tf.float32) / 255.0
            policy, _ = self.model(state)
            action = self.prob(policy)
            action = np.squeeze(action)
        else:
            action = self.env.action_space.sample()
        return action

    def update(self, state, next_state, reward, done, action):
        sample_range = np.arange(self.rollout)
        np.random.shuffle(sample_range)
        sample_idx = sample_range[:self.batch_size]

        state = [state[i] for i in sample_idx]
        next_state = [next_state[i] for i in sample_idx]
        done = [done[i] for i in sample_idx]
        action = [action[i] for i in sample_idx]

        variables = self.model.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(variables)
            _, current_value = self.model(tf.convert_to_tensor(state, dtype=tf.float32))
            _, next_value = self.model(tf.convert_to_tensor(next_state, dtype=tf.float32))
            current_value, next_value = tf.squeeze(current_value), tf.squeeze(next_value)
            target = tf.stop_gradient(self.gamma * (1 - tf.convert_to_tensor(done, dtype=tf.float32)) * next_value + tf.convert_to_tensor(reward, dtype=tf.float32))
            value_loss = tf.reduce_mean(tf.square(target - current_value) * 0.5)

            policy, _ = self.model(tf.convert_to_tensor(state, dtype=tf.float32))
            entropy = tf.reduce_mean(-policy * tf.math.log(policy + 1e-8)) * 0.1
            action = tf.convert_to_tensor(action, dtype=tf.int32)
            onehot_action = tf.one_hot(action, self.action_size)
            action_policy = tf.reduce_sum(onehot_action * policy, axis=1)
            advantages = tf.stop_gradient(target - current_value)
            pi_loss = -tf.reduce_mean(tf.math.log(action_policy + 1e-8) * advantages) - entropy

            total_loss = pi_loss + value_loss

        grads = tape.gradient(total_loss, variables)
        self.optimizer.apply_gradients(zip(grads, variables))

    def plot(self, rewards, mean_rewards):
        plt.title('BattleZone-ram-v0 A2C')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(rewards, 'C0')
        # Take 100 episode averages and plot them too
        if len(rewards) >= 100:
            plt.plot(mean_rewards, 'C1')

        plt.savefig('%d.png' % ID)

    def run(self):
        observation = self.env.reset()
        episode = 0
        score = 0
        scores = []
        mean_scores = [0] * 99

        frames = []
        best_score = 0

        timestamp = time.time()

        while True:
            states, next_states = [], []
            rewards, dones, actions = [], [], []

            for _ in range(self.rollout):
                action = self.get_action(observation)
                next_state, reward, done, info = self.env.step(action)
                self.env.render()
                frames.append(self.env.render(mode='rgb_array'))

                score += reward
                done = info['ale.lives'] == 4 or done

                states.append(observation)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                actions.append(action)

                observation = next_state

                if done:
                    episode += 1
                    timestamp_ = int(time.time() - timestamp)
                    args = (timestamp_ // 3600, (timestamp_ % 3600 // 60), timestamp_ % 60, episode, score)
                    print('[%02dh %02dm %02ds] Episode %04d: %f' % args)
                    scores.append(score)
                    if best_score < score:
                        best_score = score
                        save_gif(frames, path='', filename='%d.gif' % ID)
                    score = 0
                    frames = []

                    if len(scores) >= 100:
                        mean_scores.append(np.mean(scores[-100:]))

                    if episode % 20 == 0:
                        self.plot(scores, mean_scores)

                        # Update epsilon
                        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

                    if episode % 100 == 0:
                        self.model.save_weights('%d.h5' % ID)

                    observation = self.env.reset()

            self.update(state=states, next_state=next_states, reward=rewards, done=dones, action=actions)


"""
class ActorCritic(tf.keras.Model):

    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(4, 4), input_shape=input_shape)
        self.conv2 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(3, 3))
        self.conv3 = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2))
        self.lstm = tf.keras.layers.LSTM(16)
        self.value_head = tf.keras.layers.Dense(1, name='value')
        self.action_head = tf.keras.layers.Dense(n_actions, activation='softmax', name='action')

    def call(self, inputs, training=False):
        x = tf.keras.activations.relu(tf.keras.layers.BatchNormalization()(self.conv1(inputs)))
        x = tf.keras.activations.relu(tf.keras.layers.BatchNormalization()(self.conv2(x)))
        x = tf.keras.activations.relu(tf.keras.layers.BatchNormalization()(self.conv3(x)))
        x = self.lstm(tf.expand_dims(tf.keras.layers.Flatten()(x), axis=0))
        value_logit = self.value_head(x)
        action_logit = self.action_head(x)
        return value_logit, action_logit
"""


def save_gif(frames, path='', filename='a.gif'):
    plt.figure(figsize=(frames[0].shape[1], frames[0].shape[0]), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


if __name__ == "__main__":
    # Best 100-episode average reward was 8330.00 ± 802.94.
    # Total runtime: 18h
    env = gym.make('BattleZone-ram-v0')
    Agent(env).run()
