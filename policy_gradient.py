""" implements a simple policy gradient (actor critic technically) agent """

import argparse
import gym
import time
from gym.spaces import Discrete
import numpy as np
from scipy.signal import lfilter
from scipy.misc import imsave, imresize
import tensorflow as tf
import tensorflow.contrib.slim as slim

parser = argparse.ArgumentParser(description=None)
parser.add_argument('-e', '--env', default='Breakout-v3', type=str, help='gym environment')
parser.add_argument('-b', '--batch_size', default=10000, type=int, help='batch size to use during learning')
parser.add_argument('-l', '--learning_rate', default=1e-3, type=float, help='used for Adam')
parser.add_argument('-g', '--discount', default=0.99, type=float, help='reward discount rate to use')
parser.add_argument('-n', '--hidden_size', default=20, type=int, help='number of hidden units in net')
parser.add_argument('-c', '--gradient_clip', default=40.0, type=float, help='clip at this max norm of gradient')
parser.add_argument('-v', '--value_scale', default=0.5, type=float, help='scale of value function regression in loss')
parser.add_argument('-t', '--entropy_scale', default=0, type=float, help='scale of entropy penalty in loss')
parser.add_argument('-m', '--max_steps', default=10000, type=int, help='max number of steps to run for')
args = parser.parse_args()
print(args)

# -----------------------------------------------------------------------------
def process_frame(frame):
    """ Atari specific preprocessing, consistent with DeepMind """
    reshaped_screen = frame.astype(np.float32).mean(2)      # grayscale
    resized_screen = imresize(reshaped_screen, (84, 110)) # downsample
    x = resized_screen[18:102, :]                           # crop top/bottom
    x = imresize(x, (42, 42)).astype(np.float32)                             # downsample
    x *= (1.0 / 255.0)                                      # place in [0,1]
    x = np.reshape(x, [42, 42, 1])                          # introduce channel
    return x

def policy_spec(x):
  net = slim.conv2d(x, args.hidden_size, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.elu, scope='conv1')
  net = slim.conv2d(net, args.hidden_size, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.elu, scope='conv2')
  net = slim.flatten(net)
  action_logits = slim.fully_connected(net, num_actions, activation_fn=None, scope='fc_act')
  value_function = slim.fully_connected(net, 1, activation_fn=None, scope='fc_value')
  return action_logits, value_function

def rollout(n, max_steps_per_episode=4500):
  """ gather a single episode with current policy """

  observations, actions, rewards, discounted_rewards = [], [], [], []
  ob = env.reset()
  ep_steps = 0
  num_episodes = 0
  ep_start_pointer = 0
  prev_obf = None
  while True:

    # we concatenate the previous frame to get some motion information
    obf_now = process_frame(ob)
    obf_before = obf_now if prev_obf is None else prev_obf
    obf = np.concatenate((obf_before, obf_now), axis=2)
    #obf = obf_now - obf_before
    prev_obf = obf_now

    # run the policy
    action = sess.run(action_index, feed_dict={x: np.expand_dims(obf, 0)}) # intro a batch dim
    action = action[0][0] # strip batch and #of samples from tf.multinomial

    # execute the action
    ob, reward, done, info = env.step(action)
    ep_steps += 1

    observations.append(obf)
    actions.append(action)
    rewards.append(reward)

    if done or ep_steps >= max_steps_per_episode:
      num_episodes += 1
      ep_steps = 0
      prev_obf = None
      discounted_rewards.append(discount(rewards[ep_start_pointer:], args.discount))
      ep_start_pointer = len(rewards)
      ob = env.reset()
      if len(rewards) >= n: break

  return np.stack(observations), np.stack(actions), np.stack(rewards), np.concatenate(discounted_rewards), {'num_episodes':num_episodes}

def discount(x, gamma): 
  return lfilter([1],[1,-gamma],x[::-1])[::-1]
# -----------------------------------------------------------------------------

# create the environment
env = gym.make(args.env)
num_actions = env.action_space.n

# compile the model
x = tf.placeholder(tf.float32, (None,) + (42,42,2), name='x')
action_logits, value_function = policy_spec(x)
action_index = tf.multinomial(action_logits - tf.reduce_max(action_logits, 1, keep_dims=True), 1) # take 1 sample
# compile the loss: 1) the policy gradient
sampled_actions = tf.placeholder(tf.int32, (None,), name='sampled_actions')
discounted_reward = tf.placeholder(tf.float32, (None,), name='discounted_reward')
pg_loss = tf.reduce_mean((discounted_reward - value_function) * tf.nn.sparse_softmax_cross_entropy_with_logits(action_logits, sampled_actions))
# and 2) the baseline (value function) regression piece
value_loss = args.value_scale * tf.reduce_mean(tf.square(discounted_reward - value_function))
# and 3) entropy regularization
action_log_prob = tf.nn.log_softmax(action_logits)
entropy_loss = -args.entropy_scale * tf.reduce_sum(action_log_prob*tf.exp(action_log_prob))
# add up and minimize
loss = pg_loss + value_loss + entropy_loss
# create the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
grads = tf.gradients(loss, tf.trainable_variables())
grads, _ = tf.clip_by_global_norm(grads, args.gradient_clip) # gradient clipping
grads_and_vars = list(zip(grads, tf.trainable_variables()))
train_op = optimizer.apply_gradients(grads_and_vars)

# tf init
sess = tf.Session()
sess.run(tf.initialize_all_variables())
n = 0
mean_rewards = []
while n <= args.max_steps: # loop forever
  n += 1

  # collect a batch of data from rollouts and do forward/backward/update
  t0 = time.time()
  observations, actions, rewards, discounted_reward_np, info = rollout(args.batch_size)
  t1 = time.time()
  sess.run(train_op, feed_dict={x:observations, sampled_actions:actions, discounted_reward:discounted_reward_np})
  t2 = time.time()

  average_reward = np.sum(rewards)/info['num_episodes']
  mean_rewards.append(average_reward)
  print('step %d: collected %d frames in %fs, mean episode reward = %f (%d eps), update in %fs' % \
        (n, observations.shape[0], t1-t0, average_reward, info['num_episodes'], t2-t1))

print(args)
print('total average reward: %f +/- %f (min %f, max %f)' % \
      (np.mean(mean_rewards), np.std(mean_rewards), np.min(mean_rewards), np.max(mean_rewards)))
