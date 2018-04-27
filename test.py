import numpy as np
from class2048 import game2048 as g2048


env = g2048()

a = np.array([[0,0,1,1],
              [0,2,3,20],
              [2,2,1,1],
              [0,1,1,0]])

env._set_state(np.reshape(a,(4,4)))

env.render()


# env._add_random_tiles()
# print(env.get_state())
# env._move(4)
# print(env.get_state())

#print(env._condense(np.array([0,2,1,1])))
