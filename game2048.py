import numpy as np
import random
import sys

np.set_printoptions(suppress=True)


class game2048(object):
    '''
    Author: David Kaplan
    Class to implement the game 2048
    Default size is a 4x4 grid, but you can change the size with the parameter `size`
    THE VALUES AT THE BOXES ARE __THE__ POWER OF 2, not 2^the_power
    In the first iteration, render is a print function, not an actual picture window

    Iteration 1: April 27, 2018
        * Render is just a print statement
    '''
    def __init__(self,
                 size=4,
                 dodebug=False,
                 P_1_new_tile = 0.8, #probability that there is 1 new tile generated
                 P_val_2_generated = 0.9): #probability the new value of the tile is 2
        '''
        states:
            row (dim 1)
        col  1  2  3  4
             2
             3
             4
        Default size is 4
        '''
        self.state = np.zeros(shape=(size,size))
        self.size = size
        #1: up, 2: down, 3: left, 4: right
        self.action_space = np.array([1,2,3,4])

        self.dodebug = dodebug

        self.did_reset = False

        #Probability of 1 tile generated = P_1_new_tile
        #Probability of 2 tiles generated = 1 - P_1_new_tile
        self.P_1_new_tile = P_1_new_tile
        #Probability of tile value 2 is generated = P_val_2_generated
        #Probability of tile value 4 is generated = 1 - P_val_2_generated
        self.P_val_2_generated = P_val_2_generated

    def reset(self):
        self.state = np.zeros(shape=(self.size,self.size))
        self._add_random_tiles()
        self.did_reset = True

    def step(self,action):
        '''
        Returns 4 variables:
         - observations (2-dim array): we just return the 2048 grid
         - reward (float): how much reward did we get for the past step
         - done (boolean): Did we run out of spots? If yes then return True
         - info (dict): ___TODO___ *****************************
        '''
        if not self.did_reset:
            print('Did not reset before taking a step')
            sys.exit()
        reward = self._move(action)

        done = True if len(np.where(self.states.flatten() == 0)[0]) == 0 else False

        return self.state,reward,done,0

    def render(self):
        temp = np.zeros(shape=(self.size,self.size))
        for row in range(self.size):
            for col in range(self.size):
                if self.state[row,col] == 0:
                    temp[row,col] = 0
                else:
                    temp[row,col] = 2 ** self.state[row,col]
        print(np.matrix(temp))

    def _set_state(self,arr):
        self.state = arr

    def get_state(self):
        return self.state

    def get_size(self):
        return self.size

    def _move(self,action,add_rand=False):
        '''
        Returns the new state and reward after taking action `action`.
        Uses the function `_condense` to condense each row or column
        '''
        cumm_reward = 0
        #up
        if action == 1:
            for i in range(self.size):
                self.state[:,i],r = self._condense(self.state[:,i])
                cumm_reward += r
        #down
        if action == 2:
            for i in range(self.size):
                a,r = self._condense(np.fliplr([self.state[:,i]])[0])
                self.state[:,i] = np.fliplr([a])[0]
                cumm_reward += r

        #left
        if action == 3:
            for i in range(self.size):
                self.state[i,:],r = self._condense(self.state[i,:])
                cumm_reward += r

        #right
        if action == 4:
            for i in range(self.size):
                a,r = self._condense(np.fliplr([self.state[i,:]])[0])
                self.state[i,:] = np.fliplr([a])[0]
                cumm_reward += r
        return cumm_reward

    def _condense(self,arr):
        '''
        Input is a 1-dim array of either a column or a row
        [x_1,x_2,...,x_size]
        returns a np array of size `size` that is condensed to the side of x_1
        '''
        #return array
        ret = np.array([])
        reward = 0
        #First, get rid of all zeros in arr
        _arr=np.array([])
        for a in arr:
            if a != 0:
                _arr=np.append(_arr,a)
        #combine neighbouring boxes of _arr
        i = 0
        while i < len(_arr):
            if i < len(_arr)-1:
                if _arr[i] == _arr[i+1]:
                    ret = np.append(ret,_arr[i]+1)
                    #only add reward if tiles combine
                    reward += 2**(ret[-1])
                    i += 1
                else:
                    ret = np.append(ret,_arr[i])
            else:
                ret = np.append(ret,_arr[i])
            i += 1
        # make ret the right length by zero padding
        if len(ret) < self.size:
            ret = np.append(ret,np.zeros(self.size-len(ret)))
        return ret,reward

    def _add_random_tiles(self):
        '''
        After you move, new tiles get displayed in the empty space.
        Probabilities of the number of new tiles generated and the values
        that are generated are set in the constructor
        '''
        #temporary state variable
        _state = self.state.flatten()
        #find indecies where it is empty (value = 0)
        indecies = np.where(_state == 0)[0]

        # generate the number of tiles to do
        # check if there are enough spaces
        num_tiles = 1 if random.random() < self.P_1_new_tile and len(indecies) == 1 else 2

        prev_index = -1
        for i in range(num_tiles):
            #generate value to set
            val = 1 if random.random() < self.P_val_2_generated else 2
            #generate index and then set
            ind = indecies[random.randint(0,len(indecies)-1)]
            _state[ind] = val
        #set self.state to reshaped _state
        self.state = np.reshape(_state,(self.size,self.size))
