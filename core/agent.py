import tensorflow as tf
from .layer import *

class Agent(object):
    def __init__(self,architecture,activations):
        self.arch = architecture
        self.act = activations
        self.depth = len(architecture)-1
        self.layers = [None]*self.depth
        self.logits = [None]*self.depth

        self._build_variable()
        self.action_op = self._build_agent()
        self.learn_op = self._get_learn_op()

    def _build_variable(self):
        self.temperature = tf.placeholder('float')
        for l in range(self.depth):
            self.layers[l] = Dense(input_dim=self.arch[l],output_dim=self.arch[l+1],activation=self.act[l],name='fc_%d'%l)

    def _build_agent(self):
        self.observation = tf.placeholder('float',[None,28*28])
        self.logits[0] = self.layers[0](self.observation)
        for l in range(1,self.depth):
            self.logits[l] = self.layers[l](self.logits[l-1])
        gumble_softmax = tf.contrib.distributions.RelaxedOneHotCategorical
        act = gumble_softmax(self.temperature,logits = self.logits[-1])
        return act.sample()

    def _get_learn_op(self):
        self.reward = tf.placeholder('float')
        self.act_played = tf.placeholder('float',[None,10])
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.act_played,logits=self.logits[-1])*self.reward
        opt = tf.train.AdamOptimizer()
        return opt.minimize(self.loss)