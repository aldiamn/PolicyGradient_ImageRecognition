# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 10:44:21 2017

@author: tingan
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class MNIST_Env(object):
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.mnist = input_data.read_data_sets("data/", one_hot=True)

    def get_observation(self):
        self.current_batch_x, self.current_batch_y = self.mnist.train.next_batch(self.batch_size)
        return self.current_batch_x

    def get_ans(self):
        return self.current_batch_y

    def get_iter_per_epoch(self):
        return int(self.mnist.train.num_examples/self.batch_size)    

    def get_reward_op(self):
        self.agent_action = tf.placeholder(shape = [None],dtype = 'int32')
        self.correct_ans = tf.placeholder(shape = [None,10],dtype = 'int32')
        correct_prediction = tf.equal(self.agent_action, tf.cast(tf.argmax(self.correct_ans, 1),'int32'))
        reward = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        reward = 2*(reward-0.5)
        return reward

    def generate_reward(self,agent_action):
        correct_prediction = np.equal(agent_action,np.argmax(self.current_batch_y,axis=1)).astype(float)
        reward = np.mean(correct_prediction,axis=0)
        reward = 2*(reward-0.5)
        return reward

    def generate_accuracy(self,agent_action):
        correct_prediction = np.equal(agent_action,np.argmax(self.current_batch_y,axis=1)).astype(float)
        acc = np.mean(correct_prediction,axis=0)
        return acc

    def get_test_observation(self):
        self.current_batch_x, self.current_batch_y = self.mnist.test.images, self.mnist.test.labels
        return self.current_batch_x