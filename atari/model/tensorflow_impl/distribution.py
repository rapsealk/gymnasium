#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import tensorflow as tf


class ProbabilityDistribution(tf.keras.Model):
    def call(self, logits, **kwargs):
        # Sample a random categorical action from the given logits.
        return tf.random.categorical(logits, 1)
