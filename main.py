import tensorflow as tf
from core.mnist_env import MNIST_Env
from core.agent import Agent

epoch = 10
batch_size = 1
temperature = 1.0
tmp_decay = True
tmp_decay_epoch = 10
display_epoch = 1

env = MNIST_Env(batch_size=batch_size)
architecture = [28*28,100,100,10]
activation = [tf.nn.sigmoid,tf.nn.sigmoid,None]
chris = Agent(architecture=architecture,activations=activation)
#reward_op = env.get_reward_op()
init = tf.global_variables_initializer()
episode = env.get_iter_per_epoch()

with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch):
        reward_avg = 0
        for i in range(episode):
            obv = env.get_observation()
            #ans = env.get_ans()
            action = sess.run(chris.action_op,feed_dict={chris.observation:obv,chris.temperature:temperature})
            #reward = sess.run(reward_op,feed_dict={env.agent_action:action,env.correct_ans:ans})
            reward = env.generate_reward(action)
            _ = sess.run(chris.learn_op,feed_dict={chris.act_played:action, 
                                                   chris.reward:reward,
                                                   chris.observation:obv,
                                                   chris.temperature:temperature})
            reward_avg += reward
        reward_avg = reward_avg/episode
        if (ep+1)%tmp_decay_epoch == 0 and tmp_decay:
            temperature=temperature*0.8
        if (ep+1)%display_epoch == 0:
            print('Epoch:%04d, reward:%.3f'%(ep+1,reward_avg))
    
    #test agent
    obv_test = env.get_test_observation()
    action_test = sess.run(chris.action_op,feed_dict={chris.observation:obv_test,chris.temperature:temperature})
    acc = env.generate_accuracy(action_test)
    print('Test Accuracy: %.3f'%acc)    