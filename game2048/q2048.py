import numpy as np
import tensorflow as tf
from game2048 import Game2048
import os
from plotting import plot_episode_stats
from Estimator import Estimator,deep_q_learning

env = Game2048()

tf.reset_default_graph()

# Where we save our checkpoints and graphs
experiment_dir = os.path.abspath("./experiments/")

# Create a glboal step variable
global_step = tf.Variable(0, name='global_step', trainable=False)

# Create estimators
q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")


# Run the experiment
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t, stats in deep_q_learning(sess,
                                    env,
                                    q_estimator=q_estimator,
                                    target_estimator=target_estimator,
                                    experiment_dir=experiment_dir,
                                    num_episodes=50000,
                                    replay_memory_size=50000,
                                    replay_memory_init_size=50000,
                                    update_target_estimator_every=10000,
                                    epsilon_start=1.0,
                                    epsilon_end=0.01,
                                    epsilon_decay_steps=50000,
                                    discount_factor=0.99,
                                    batch_size=32,
                                    force_legal_move=True):

        print("\nEpisode Reward: {}".format(stats.episode_rewards[-1]))
    plot_episode_stats(stats)

#print greedy run
ACTION_NAMES = ['up','down','left','right']
data = np.genfromtxt("greedy_run.csv",delimiter=",")
for i in range(len(data[:,0])):
    state = np.reshape(data[i,0:16],[4,4])
    action = data[i,-1]
    print('state:\n{}\naction taken: {}'.format(state,ACTION_NAMES[action]))
