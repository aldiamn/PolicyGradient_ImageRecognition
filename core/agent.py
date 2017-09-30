import tensorflow as tf
from layers import *

def Agent(object):
    def __init__(self,architecture,activations):
        self.arch = architecture
        self.act = activations
        self.depth = len(architecture)-1
        self.layers = [None]*self.depth
        self.logits = [None]*self.depth

        self._build_variable()
        self.action_op = self._build_agent()

    def _build_variable(self):
        for l in range(self.depth):
            self.layers[l] = Dense(input_dim=self.arch[l],output_dim=self.arch[l+1],activation=self.act[l],name='fc_%d'%l)

    def _build_agent(self):
        self.observation = tf.placeholder('float',[None,28*28])
        self.logits[0] = self.layers[0](self.observation)
        for l in range(1,self.depth):
            self.logits[l] = self.layers[l](self.logits[l-1])

    def get_learn_op(self):
        