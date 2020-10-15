#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


class DQN(tf.keras.Model):
    def __init__(self, n):
        super(DQN, self).__init__()
        self.hidden1 = tf.keras.layers.Conv2D(16, kernel_size=(8, 8), strides=(4, 4), input_shape=(84, 84, 4))
        self.hidden2 = tf.keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2, 2))
        self.hidden3 = tf.keras.layers.Dense(256)
        self.hidden4 = tf.keras.layers.Dense(n)

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)

    def call(self, inputs):
        x = tf.nn.relu(self.hidden1(inputs))
        x = tf.nn.relu(self.hidden2(x))
        x = tf.nn.relu(self.hidden3(x))
        return self.hidden4(x)

    def get_pi(self, inputs):
        return tf.nn.softmax(self(inputs))

    def preprocess(self, observation):
        assert observation.shape == (210, 160, 3)
        observation = tf.image.rgb_to_grayscale(observation)
        observation = tf.image.resize(observation, (110, 84))

        observation = np.asarray(observation)
        observation = np.squeeze(observation)
        observation = observation[-84:, :]

        assert observation.shape == (84, 84)

        return observation.astype(np.float32)

    def train(self, batch, gamma=0.99):
        s = np.array([b[0] for b in batch])
        a = np.array([b[1] for b in batch])
        r = np.array([b[2] for b in batch])
        s_ = np.array([b[3] for b in batch])
        dones = np.array([b[4] for b in batch])

        # target_network = DQN()
        # target_network.set_weights(self.get_weights())

        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            """
            q = self(s)[:, 0, 0]
            # q_a = tf.gather(q, a)  # (32, 4)
            # print('q_a.shape:', q_a.shape)
            q = tf.gather(q, a, axis=1)
            q_ = self(s_)[:, 0, 0]
            q_a_ = tf.math.argmax(q_, axis=-1)
            q_ = tf.gather(q_, q_a_, axis=1)
            target = r + gamma * q_ * dones
            loss = tf.reduce_mean(tf.square(q - target)) * 0.5
            """

            target = r + gamma *    # yi
            loss = 0.5 * tf.reduce_mean(tf.math.square(target - q))

        grads = tape.gradient(loss, self.trainable_variables)
        # grads, _ = tf.clip_by_global_norm(grads, 40.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss


if __name__ == "__main__":
    pi = DQN(4)
    observation = np.random.randint(0, 256, size=(1, 84, 84, 4))
    policy = pi(observation.astype(np.float32))
    policy = np.squeeze(policy)
    # print('policy:', policy, policy.shape)
    print(np.sum(policy), np.sum(policy[0, 0]))
    print('policy:', policy[0, 0])
    print(np.argmax(policy, axis=2))
