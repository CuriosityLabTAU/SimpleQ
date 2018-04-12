import numpy as np
import random
import operator
import matplotlib.pyplot as plt
import copy

class World:
    def __init__(self):
        self.board_height = 1
        self.board_width = 20
        self.board = np.zeros((self.board_height, self.board_width))
        self.goal = np.ones((self.board_height, self.board_width))
        self.initial_pos = (0,0)
        self.current_pos = (0,0)

    def set_board_size(self, height, width):
        self.board_height = height
        self.board_width = width
        self.board = np.zeros((self.board_height, self.board_width))
        self.goal = np.ones((self.board_height, self.board_width))

    def clean_board(self):
        self.board = np.zeros((self.board_height, self.board_width))

    # def get_current_state(self):
    #     if self.current_pos[0] == 0:
    #         up = -1
    #     elif self.board[self.current_pos[0]-1, self.current_pos[1]] == 1:
    #         up = 1
    #     else:
    #         up = 0
    #     if self.current_pos[0] == self.board_height-1:
    #         down = -1
    #     elif self.board[self.current_pos[0]+1, self.current_pos[1]] == 1:
    #         down = 1
    #     else:
    #         down = 0
    #     if self.current_pos[1] == 0:
    #         left = -1
    #     elif self.board[self.current_pos[0], self.current_pos[1]-1] == 1:
    #         left = 1
    #     else:
    #         left = 0
    #     if self.current_pos[1] == self.board_width-1:
    #         right = -1
    #     elif self.board[self.current_pos[0], self.current_pos[1] + 1] == 1:
    #         right = 1
    #     else:
    #         right = 0
    #     return (self.current_pos[0], self.current_pos[1], up, right, down, left)

    def get_current_state(self):
        if self.current_pos[0] == 0:
            up = -1
        elif self.board[self.current_pos[0]-1, self.current_pos[1]] == 1:
            up = -1
        else:
            up = 0
        if self.current_pos[0] == self.board_height-1:
            down = -1
        elif self.board[self.current_pos[0]+1, self.current_pos[1]] == 1:
            down = -1
        else:
            down = 0
        if self.current_pos[1] == 0:
            left = -1
        elif self.board[self.current_pos[0], self.current_pos[1]-1] == 1:
            left = -1
        else:
            left = 0
        if self.current_pos[1] == self.board_width-1:
            right = -1
        elif self.board[self.current_pos[0], self.current_pos[1] + 1] == 1:
            right = -1
        else:
            right = 0
        return (self.current_pos[0], self.current_pos[1], up, right, down, left)

    def get_cord_state(self, i, j):
        if i == 0:
            up = -1
        elif self.board[i-1, j] == 1:
            up = -1
        else:
            up = 0
        if i == self.board_height-1:
            down = -1
        elif self.board[i+1, j] == 1:
            down = -1
        else:
            down = 0
        if j == 0:
            left = -1
        elif self.board[i, j-1] == 1:
            left = -1
        else:
            left = 0
        if j == self.board_width-1:
            right = -1
        elif self.board[i, j + 1] == 1:
            right = -1
        else:
            right = 0
        return (i, j, up, right, down, left)


    def set_current_pos(self, i, j):
        self.current_pos = (i,j)
        self.board[i,j] = 1 # paint the current position

    # def set_random_pos(self):
    #     i = random.choice(range(self.board_height))
    #     j = random.choice(range(self.board_width))
    #     self.current_pos = (i,j)
    #     self.board[i, j] = 1  # paint the current position

    def set_random_pos(self):
        white_board = np.where(self.board==0)  # indexes of white board
        if len(white_board[0]) != 0:
            rnd_index = random.choice(range(len(white_board[0])))
            i = white_board[0][rnd_index]
            j = white_board[1][rnd_index]
            self.current_pos = (i,j)
            self.board[i, j] = 1  # paint the current position

    def get_actions(self, state):
        actions = []
        if state[2] != -1:
            actions.append((-1,0)) # go up
        if state[3] != -1:
            actions.append((0, 1)) # go right
        if state[4] != -1:
            actions.append((1,0))  # go down
        if state[5] != -1:
            actions.append((0, -1)) # go left
        return actions

    def update_state(self, action):
        state = self.get_current_state()
        actions = self.get_actions(state)
        if action not in actions:
            print ('illegal action.', state, action)
        else:
            next_pos = tuple(map(operator.add, self.current_pos, action))
#            if self.board[next_pos] == 0:
#                reward = 10
#            else:
#                reward = -10

            #reward = Learner(state[1], action[1], next_pos[1])
            reward = 1
            print('current_state = ', state[1],'action = ', action[1], 'next_state = ', next_pos[1] )
            self.current_pos = next_pos
            # self.board[(self.current_pos)] = 1
            # if np.array_equal(self.board, self.goal):
            #     reward += 1000
        return self.get_current_state(), reward

    def is_goal_achieved(self):
        if np.array_equal(self.board, self.goal):
            return True
        else:
            return False

    def display(self, title=''):
        current = np.zeros((self.board_height, self.board_width))
        current[self.current_pos]=3
        plt.imshow(self.board+current, interpolation='none')
        plt.title(title)
        plt.pause(0.01)
