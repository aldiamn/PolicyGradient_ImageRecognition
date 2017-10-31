import tensorflow as tf
from core.mnist_env import MNIST_Env
from core.agent import Agent

epoch = 100
batch_size = 100

env = MNIST_Env(batch_size=batch_size)
architecture = [28*28,100,100,10]
activation = [tf.nn.relu,tf.nn.relu,None]
chris = Agent(architecture=architecture,activations=activation)
reward_op = env.get_reward_op()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        obv = env.get_observation()
        ans = env.get_ans()
        action = sess.run(chris.action_op,feed_dict={chris.observation:obv})
        reward = sess.run(reward_op,feed_dict={env.agent_action:action,env.correct_ans:ans})
        print('Epoch:%03d, reward:%.3f'%(i+1,reward))
        _ = sess.run(chris.learn_op,feed_dict={chris.act_played:action, chris.reward:reward,chris.observation:obv})

        