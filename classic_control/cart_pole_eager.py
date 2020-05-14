#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, ReLU


class DQN(tf.keras.Model):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = Conv2D(16, kernel_size=5, strides=(2, 2))
        self.bn1 = BatchNormalization(momentum=0.1)
        self.conv2 = Conv2D(32, kernel_size=5, strides=(2, 2))
        self.bn2 = BatchNormalization(momentum=0.1)
        self.conv3 = Conv2D(32, kernel_size=5, strides=(2, 2))
        self.bn3 = BatchNormalization(momentum=0.1)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convw * convh * 32
        self.head = Dense(linear_input_size)

    def call(self, x):
        x = ReLU(self.bn1(self.conv1(x)))
        x = ReLU(self.bn2(self.conv2(x)))
        x = ReLU(self.bn3(self.conv3(x)))
        return self.head(x)


if __name__ == "__main__":
    pass
