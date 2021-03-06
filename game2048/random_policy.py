import numpy as np
from game2048 import Game2048 as G2048
import random
import time
import matplotlib.pyplot as plt

env = G2048()
env.reset()


moves = [0,1,2,3]
history = np.array([])
start = time.time()
# Random Policy
for i in range(500):
    print(i)
    env.reset()
    cumm_reward = 0
    done = False

    while not done:
        action = random.choice(moves)
        obs,reward,done,_ = env.step(action)
        cumm_reward += reward
    history = np.append(history,cumm_reward)

history = np.sort(history)
print('mean score', np.mean(history), 'stddev score', np.std(history))
print('high score', history[-1], 'low score', history[0])
print(time.time() - start)

plt.plot(history)
plt.show()
