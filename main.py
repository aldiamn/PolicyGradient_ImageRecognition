import tensorflow as tf
from core.mnist_env import MNIST_Env
from core.agent import Agent

epoch = 100
batch_size = 1
temperature = 1.0
display_epoch = 10

env = MNIST_Env(batch_size=batch_size)
architecture = [28*28,100,100,10]
activation = [tf.nn.relu,tf.nn.relu,None]
chris = Agent(architecture=architecture,activations=activation)
reward_op = env.get_reward_op()
init = tf.global_variables_initializer()
episode = env.get_iter_per_epoch()

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch):
        reward_avg = 0
        for i in range(episode):
            obv = env.get_observation()
            ans = env.get_ans()
            action = sess.run(chris.action_op,feed_dict={chris.observation:obv,chris.temperature:temperature})
            reward = sess.run(reward_op,feed_dict={env.agent_action:action,env.correct_ans:ans})
            _ = sess.run(chris.learn_op,feed_dict={chris.act_played:action, 
                                                   chris.reward:reward,
                                                   chris.observation:obv,
                                                   chris.temperature:temperature})
            reward_avg += reward
            #print(action)
        reward_avg = reward_avg/episode
        if (ep+1)%display_epoch == 0:
            temperature=temperature*0.8
            print('Epoch:%04d, reward:%.3f'%(ep+1,reward_avg))
        