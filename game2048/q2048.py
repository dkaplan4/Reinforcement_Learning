'''
Implementing q-learning for 2048
'''
import numpy as np
import collections
import math
from game2048 import Game2048 as G2048
import tensorflow as tf

env = G2048()

# Maximum value for this implementation = 2^15 = 32,768


# Hyperparameters
buckets = 0
n_episodes=5000
min_alpha=0.1  # learning rate
min_epsilon=0.1  # exploration rate
gamma=0.8  # discount factor
ada_divisor=25 #scalar we are multiplying the alpha curves


def run_episode:
    return 0