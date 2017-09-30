# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 10:44:21 2017

@author: tingan
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class MNIST_Env(object):
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.mnist = input_data.read_data_sets("data/", one_hot=True)

    def get_observation(self):
        self.current_batch_x, self.current_batch_y = self.mnist.next_batch(self.batch_size)
        return self.current_batch_x

    def get_reward_op(self):
        self.agent_action = tf.placeholder(shape = [None,10],dtype = int32)
        correct_prediction = tf.equal(tf.argmax(self.agent_action, 1), tf.argmax(self.current_batch_y, 1))
        reward = tf.reduce_sum(tf.cast(correct_prediction, "float"))
        return reward