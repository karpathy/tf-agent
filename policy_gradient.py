""" implements a simple policy gradient agent """

import argparse
import cv2
import gym
import time
from gym.spaces import Discrete
import numpy as np
from scipy.signal import lfilter
import tensorflow as tf
import tensorflow.contrib.slim as slim

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='Breakout-v3', type=str, help='gym environment')
parser.add_argument('-b', '--batch_size', default=10000, type=int, help='batch size to use during learning')
parser.add_argument('-l', '--learning_rate', default=1e-3, type=float, help='used for Adam')
parser.add_argument('-g', '--discount', default=0.99, type=float, help='reward discount rate to use')
parser.add_argument('-n', '--hidden_size', default=24, type=int, help='number of hidden units in net')
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
  conv1 = slim.conv2d(x, args.hidden_size, [5, 5], stride=2, padding='SAME', scope='conv1')
  conv2 = slim.conv2d(conv1, num_actions, [5, 5], stride=2, padding='SAME', activation_fn=None, scope='conv2')
  action_logits = tf.reduce_mean(conv2, [1,2]) # average pool across space
  return action_logits

def rollout(n, max_steps_per_episode=4500):
  """ gather a single episode with current policy """

  observations, actions, rewards = [], [], []
  ob = env.reset()
  ep_steps = 0
  num_episodes = 0
  while True:

    # run the policy
    obf = np.expand_dims(process_frame(ob), 0) # intro a batch dim
    action = sess.run(action_index, feed_dict={x: obf})
    action = action[0][0] # strip batch and #samples from tf.multinomial

    # execute the action
    ob, reward, done, info = env.step(action)
    ep_steps += 1

    observations.append(obf[0])
    actions.append(action)
    rewards.append(reward)

    if done or ep_steps >= max_steps_per_episode:
      num_episodes += 1
      ep_steps = 0
      ob = env.reset()
      if len(rewards) >= n: break

  return np.stack(observations), np.stack(actions), np.stack(rewards), {'num_episodes':num_episodes}

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
n = 0
while True: # loop forever
  n += 1

  # collect a batch of data from a rollout
  t0 = time.time()
  observations, actions, rewards, info = rollout(args.batch_size)
  
  # perform a parameter update
  t1 = time.time()
  discounted_rewards_np = standardize(discount(rewards, args.discount))
  _, loss_np = sess.run([train_op, loss], feed_dict={x:observations, sampled_actions:actions, discounted_rewards:discounted_rewards_np})
  t2 = time.time()

  print 'step %d: collected %d frames in %fs, mean episode reward = %f (%d eps), update in %fs' % \
        (n, observations.shape[0], t1-t0, np.sum(rewards)/info['num_episodes'], info['num_episodes'], t2-t1)
