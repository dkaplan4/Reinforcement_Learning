# Reinforcement Learning for 2048
 - Project by David Kaplan for CMSC389F, Spring 2018
 - Q-Learning on 2048

## Setup Notes
 - Uses Python 3
 - Matplotlib
 - Numpy
 - Tensorflow 1.8
 - Pandas
 - psutil

## Module Information
 - Gym environment: We created our own. The source code can be found in `game2048.py`. The `render()` function is just a print statement for now.


## Learning algorithms
 - Random Policy: `random_policy.py`
 - Q-Learning: `q2048.py`

## Running the algorithms

For a random policy, use `python random_policy.py`.

For DQN, use `python q2048.py`.

To see print recordings by the best performances by both DQN and random, look at
`best_run_random.txt` and `best_run_DQN.txt`.
