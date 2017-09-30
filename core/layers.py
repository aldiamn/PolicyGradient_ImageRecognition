import tensorflow as tf

def get_weight(shape,name=None,initializer=tf.contrib.layers.xavier_initializer()):
    return tf.get_variable(name=name,shape=shape,initializer=initializer)

def get_bias(shape,name=None):
    return tf.get_variable(name=name,shape=shape,initializer=tf.random_normal_initializer())

class Dense(object):
    def __init__(self,input_dim,output_dim,activation,name=None):
        with tf.variable_scope(name) as scope:
            self.weight     = get_weight(shape=[input_dim,output_dim],name='dense_weight')
            self.bias       = get_bias(shape=[output_dim],name='dense_bias')
            self.activation = activation

    def __call__(self,input_tensor):
        self.logit  = tf.matmul(input_tensor,self.weight)+self.bias
        if self.activation == None:
            self.output = self.logit
        else:
            self.output = self.activation(self.logit)
        return self.output