""" implements a simple policy gradient agent """

import argparse
import cv2
import gym
from gym.spaces import Discrete
import numpy as np
from scipy.signal import lfilter
import tensorflow as tf
import tensorflow.contrib.slim as slim

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='Breakout-v3', type=str, help='gym environment')
parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size to use during learning')
parser.add_argument('-l', '--learning_rate', default=1e-4, type=float, help='used for Adam')
parser.add_argument('-g', '--discount', default=0.99, type=float, help='reward discount rate to use')
args = parser.parse_args()

# -----------------------------------------------------------------------------
def process_frame(frame):
    """ Atari specific preprocessing, consistent with DeepMind """
    reshaped_screen = frame.astype(np.float32).mean(2)      # grayscale
    resized_screen = cv2.resize(reshaped_screen, (84, 110)) # downsample
    x = resized_screen[18:102, :]                           # crop top/bottom
    x = cv2.resize(x, (42, 42))                             # downsample
    x *= (1.0 / 255.0)                                      # place in [0,1]
    x = np.reshape(x, [42, 42, 1])                          # introduce channel
    return x

def policy_spec(x):
  conv1 = slim.conv2d(x, 10, [5, 5], stride=2, padding='SAME', scope='conv1')
  conv2 = slim.conv2d(conv1, num_actions, [5, 5], stride=2, padding='SAME', activation_fn=None, scope='conv2')
  action_logits = tf.reduce_mean(conv2, [1,2]) # average pool across space
  return action_logits

def rollout(max_steps=-1):
  """ gather a single episode with current policy """

  observations = []
  actions = []
  rewards = []
  ob = env.reset()
  done = False
  while not done and (len(rewards) < max_steps or max_steps==-1):

    # run the policy
    obf = np.expand_dims(process_frame(ob), 0) # intro a batch dim
    action = sess.run(action_index, feed_dict={x: obf}) # (batch_size, num_samples)
    action = action[0][0] # strip batch and #samples from tf.multinomial

    # execute the action
    ob, reward, done, info = env.step(action)

    observations.append(obf[0])
    actions.append(action)
    rewards.append(reward)

  return np.stack(observations), np.stack(actions), np.stack(rewards)

def discount(x, gamma): 
  return lfilter([1],[1,-gamma],x[::-1])[::-1]

def standardize(v):
  return (v-np.mean(v))/np.std(v) if v.any() else v
# -----------------------------------------------------------------------------

# create the environment
env = gym.make(args.env)
num_actions = env.action_space.n
# compile the model
x = tf.placeholder(tf.float32, (None,) + (42,42,1), name='x')
action_logits = policy_spec(x)
action_index = tf.multinomial(action_logits - tf.reduce_max(action_logits, 1, keep_dims=True), 1) # take 1 sample
# compile the loss
sampled_actions = tf.placeholder(tf.int32, (None,), name='sampled_actions')
discounted_rewards = tf.placeholder(tf.float32, (None,), name='discounted_rewards')
loss = tf.reduce_mean(discounted_rewards * tf.nn.sparse_softmax_cross_entropy_with_logits(action_logits, sampled_actions))
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
train_op = optimizer.minimize(loss)

# tf init
sess = tf.Session()
sess.run(tf.initialize_all_variables())

while True: # loop forever
  
  # collect a batch of data from a rollout
  observations, actions, rewards = rollout()
  discounted_rewards_np = standardize(discount(rewards, args.discount))

  # perform a parameter update
  for t in xrange(0, rewards.shape[0], args.batch_size):

    s = observations[t:t+args.batch_size]
    a = actions[t:t+args.batch_size]
    r = discounted_rewards_np[t:t+args.batch_size]

    _, loss_np = sess.run([train_op, loss], feed_dict={x:s, sampled_actions:a, discounted_rewards:r})

  print 'episode with %d frames, sum reward = %f' % (rewards.shape[0], np.sum(rewards))