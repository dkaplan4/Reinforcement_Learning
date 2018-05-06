import numpy as np
import random
import sys

np.set_printoptions(suppress=True)



class Game2048(object):
    '''
    Author: David Kaplan
    Class to implement the game 2048
    Default size is a 4x4 grid, but you can change the size with the parameter `size`
    THE VALUES AT THE BOXES ARE __THE__ POWER OF 2, not 2^the_power
        * 0 refers to there not being any tile at that location, not a value of 1
    In the first iteration, render is a print function, not an actual picture window

    Iteration 1: April 27, 2018
        * Render is just a print statement
    '''

    def __init__(self,
                 size=4,
                 dodebug=False,
                 penalize_wrong_moves=True,  # ends the game if 100 wrong moves are done
                 p_1_new_tile=0.8,  # probability that there is 1 new tile generated
                 p_val_2_generated=0.9):  # probability the new value of the tile is 2
        self.__state = np.zeros(shape=(size, size))
        self.__size = size
        # 0: up, 1: down, 2: left, 3: right
        self.action_space = np.array([0, 1, 2, 3])

        self.__dodebug = dodebug

        self.__did_reset = False

        self.__score = 0

        self.__num_illegal_moves = 0

        self.__total_moves = 0

        self.__penalize_wrong_moves = penalize_wrong_moves

        # Probability of 1 tile generated = P_1_new_tile
        # Probability of 2 tiles generated = 1 - P_1_new_tile
        self.__p_1_new_tile = p_1_new_tile
        # Probability of tile value 2 is generated = P_val_2_generated
        # Probability of tile value 4 is generated = 1 - P_val_2_generated
        self.__p_val_2_generated = p_val_2_generated

    def reset(self):
        self.__state = np.zeros(shape=(self.__size, self.__size))
        self.__add_random_tiles()
        self.__did_reset = True
        self.__num_illegal_moves = 0
        self.__score = 0
        self.__total_moves = 0
        return self.__state

    def step(self, action):
        '''
        Returns 4 variables:
         - observations (2-dim array): we just return the 2048 grid
         - reward (float): how much reward did we get for the past step
         - done (boolean): Did we run out of spots? If yes then return True
         - info (dict): ___TODO___ *****************************
        '''
        if not self.__did_reset:
            print('Did not reset before taking a step')
            sys.exit()
        # save the previous state
        _state = self.__state.copy()
        reward, score = self.__move(action)
        # if the state did not change, that means nothing happened and we do not add new tiles
        if not np.array_equal(_state, self.__state):
            self.__add_random_tiles()

        elif self.__penalize_wrong_moves:
            self.__num_illegal_moves += 1

        self.__score += score
        self.__total_moves += 1
        done = self.__is_game_over()
        if self.__num_illegal_moves == 100:
            done = True
        return self.__state, reward, done, 0

    def render(self):
        temp = np.zeros(shape=(self.__size, self.__size))
        for row in range(self.__size):
            for col in range(self.__size):
                if self.__state[row, col] == 0:
                    temp[row, col] = 0
                else:
                    temp[row, col] = 2 ** self.__state[row, col]
        print(np.matrix(temp))

    def get_state(self):
        return self.__state

    def get_total_steps(self):
        return self.__total_moves

    def get_best_valid_move(self,prob_actions):
        # record base state
        action_order = np.argsort(prob_actions)
        ret = []
        _state = self.__state.copy()
        for action in action_order:
            self.__move(action)
            if not np.array_equal(_state, self.__state):
                return action

        print('should not go here')
        sys.exit('get_best_valid_moves -- could not find a move? something is wrong.')

    def _set_state(self, arr):
        '''
        For Debugging only
        '''
        self.__state = arr

    def __move(self, action):
        '''
        Returns the new state and reward after taking action `action`.
        Uses the function `__condense` to condense each row or column
        '''
        cumm_reward = 0
        cumm_score = 0
        # up
        if action == 0:
            for i in range(self.__size):
                self.__state[:, i], r, s = self.__condense(self.__state[:, i])
                cumm_reward += r
                cumm_score += s
        # down
        if action == 1:
            for i in range(self.__size):
                a, r, s = self.__condense(np.fliplr([self.__state[:, i]])[0])
                self.__state[:, i] = np.fliplr([a])[0]
                cumm_reward += r
                cumm_score += s

        # left
        if action == 2:
            for i in range(self.__size):
                self.__state[i, :], r, s = self.__condense(self.__state[i, :])
                cumm_reward += r
                cumm_score += s

        # right
        if action == 3:
            for i in range(self.__size):
                a, r, s = self.__condense(np.fliplr([self.__state[i, :]])[0])
                self.__state[i, :] = np.fliplr([a])[0]
                cumm_reward += r
                cumm_score += s

        return cumm_reward, cumm_score

    def __condense(self, arr):
        '''
        Input is a 1-dim array of either a column or a row
        [x_1,x_2,...,x_size]
        returns a np array of size `size` that is condensed to the side of x_1
        '''
        # return array
        ret = np.array([])
        reward = 0
        score = 0
        # First, get rid of all zeros in arr
        _arr = np.array([])
        for a in arr:
            if a != 0:
                _arr = np.append(_arr, a)
        # combine neighbouring boxes of _arr
        i = 0
        while i < len(_arr):
            if i < len(_arr) - 1:
                if _arr[i] == _arr[i + 1]:
                    ret = np.append(ret, _arr[i] + 1)
                    # only add reward if tiles combine
                    score += 2 ** (ret[-1])
                    reward += ret[-1]
                    i += 1
                else:
                    ret = np.append(ret, _arr[i])
            else:
                ret = np.append(ret, _arr[i])
            i += 1
        # make ret the right length by zero padding
        if len(ret) < self.__size:
            ret = np.append(ret, np.zeros(self.__size - len(ret)))
        return ret, reward, score

    def __add_random_tiles(self):
        '''
        After you move, new tiles get displayed in the empty space.
        Probabilities of the number of new tiles generated and the values
        that are generated are set in the constructor
        '''
        # temporary state variable
        _state = self.__state.flatten()
        # find indexes where it is empty (value = 0)
        indexes = np.where(_state == 0)[0]

        # generate the number of tiles to do
        # check if there are enough spaces
        num_tiles = 1 if random.random() < self.__p_1_new_tile or len(indexes) == 1 else 2

        prev_index = -1
        for i in range(num_tiles):
            # generate value to set
            val = 1 if random.random() < self.__p_val_2_generated else 2
            # generate index and then set
            ind = indexes[random.randint(0, len(indexes) - 1)]
            # make sure the indexes that are generated are unique
            while ind == prev_index:
                ind = indexes[random.randint(0, len(indexes) - 1)]
            _state[ind] = val
            prev_index = ind
        # set self.__state to reshaped _state
        self.__state = np.reshape(_state, (self.__size, self.__size))

    def __is_game_over(self):
        '''
        Checks if the game is over.
        Do this by checking if any reward is returned by `__move` in all directions
        If the reward > 0, this means that there are places to move and so False is returned
        '''
        # check if the board is full
        if not bool(len(np.where(self.__state == 0)[0])):
            # record base state
            _state = self.__state
            for action in self.action_space:
                if self.__move(action)[0] > 0:
                    return False
                self.__state = _state
            return True
        else:
            return False

def get_best_valid_move(state,prob_actions):
    '''
    Returns the indexes of the moves that are possible
    '''
    env = Game2048()
    env._set_state(state)
    return env.get_best_valid_move(prob_actions)
