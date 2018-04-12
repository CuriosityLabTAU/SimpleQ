from World import *
import matplotlib.pyplot as plt
import string
import random

class Q_Learner:
    def __init__(self, world):
        self.world = world
        self.q = {}
        self.alpha = 0.9  # learning rate
        self.gamma = 0.8  # discount factor
        self.epsilon = 0.6  # epsilon greedy

    def getQ(self, state, action):
        return self.q.get((state, action), 0)  # if key doesn't exist return 0

    def updateQ(self, state1, action, reward, state2):
        if self.q.get((state1,action), None) is None:
            self.q[(state1, action)] = reward
        else:
            if self.world.get_actions(state2): # if list is not empty
                Q_max = max(self.getQ(state2, a) for a in self.world.get_actions(state2))
                cur_Q = self.getQ(state1,action)
                self.q[(state1, action)] = cur_Q + self.alpha*(reward + self.gamma*Q_max-cur_Q)

    def choose_action(self, state):
        possible_actions = self.world.get_actions(state)
        if not possible_actions: # if list is empty return None
            return None
        if random.random() > self.epsilon:
            q_list = [self.getQ(state,a) for a in possible_actions]
            maxQ = max(q_list)
            if q_list.count(maxQ) == 1:
                index = q_list.index(maxQ)
                return possible_actions[index]
            elif q_list.count(maxQ) > 1:
                options = [i for i in range(len(q_list)) if q_list[i] == maxQ]
                index = random.choice(options)
                return possible_actions[index]
        else:
            action = random.choice(possible_actions)
            return action

    def print_state(self):
        print self.world.get_current_state()

    def train(self, num_of_episodes, max_step, print_flag = False, disp_policy_flag = False, disp_policy2_flag = False, disp_flag = False, save_flag = False, image_names = 'fig'):
        steps_per_episode = []
        for k in range(num_of_episodes):
            self.world.clean_board()
            self.world.set_current_pos(0,0)  # TODO: maybe start from the same position
            for n in range(max_step):
                # Tan.print_state()
                current_state = self.world.get_current_state()
                action = self.choose_action(current_state)
                if action is None:
                    while action is None:
                        if self.world.is_goal_achieved():
                            break
                        self.world.set_random_pos()
                        current_state = self.world.get_current_state()
                        action = self.choose_action(current_state)
                if self.world.is_goal_achieved():
                    break
                next_state, reward = self.world.update_state(action)
                self.updateQ(current_state, action, reward, next_state)
                if print_flag:
                    print (n, current_state, action, reward, next_state)
                if disp_flag:
                    self.world.display()
                if self.world.is_goal_achieved():
                    #print n
                    #steps_per_episode.append(n)
                    # if disp_policy_flag:
                    #     self.print_policy(k=k, save_flag=save_flag, image_names=image_names)
                    # if disp_policy2_flag:
                    #     self.print_policy_2(k)
                    break
                if disp_policy_flag:
                    self.print_policy(k=n, save_flag=save_flag, image_names=image_names)
            steps_per_episode.append(n)
            if disp_policy_flag:
                pass
        #        self.print_policy(k=k, save_flag=save_flag, image_names=image_names)
            if disp_policy2_flag:
                pass
         #       self.print_policy_2(save_flag=save_flag, image_names=image_names)
        #print Tan.q
        return steps_per_episode

    def eval(self, num_of_episodes, max_step, print_flag=False, disp_flag=False, save_flag=False, image_names='fig'):
        frame = 0
        steps_per_episode = []
        reward_per_episode = []
        for k in range(num_of_episodes):
            current_reward=0
            self.world.clean_board()
            self.world.set_current_pos(0, 0)
            for n in range(max_step):
                current_state = self.world.get_current_state()
                action = self.choose_action(current_state)
                if action is None:
                    while action is None:
                        if self.world.is_goal_achieved():
                            break
                        self.world.set_random_pos()
                        current_state = self.world.get_current_state()
                        action = self.choose_action(current_state)

                if self.world.is_goal_achieved():
                    break
                next_state, reward = self.world.update_state(action)
                current_reward += reward
                if print_flag:
                    print (n, current_state, action, reward, next_state)
                if disp_flag:
                    self.world.display(title='Episode #' + str(k))
                    #plt.title('episode '+str(k))
                if save_flag:
                    pass
                    # for t in range(10):
                    #     plt.savefig('./figs1/' + image_names + string.zfill(str(frame), 5) + '.png', format='png')
                    #     frame += 1
                if self.world.is_goal_achieved():
                    #print n
                    # steps_per_episode.append(n)
                    break
            steps_per_episode.append(n)
            reward_per_episode.append(current_reward)
        return steps_per_episode, reward_per_episode


    def print_policy(self, k, save_flag=False, image_names='fig'):
        X = np.zeros([self.world.board_height, self.world.board_width])
        Y = np.zeros([self.world.board_height, self.world.board_width])
        # A = np.array(self.actions)
        for i in range(self.world.board_height):
            for j in range(self.world.board_width):
                state  = self.world.get_cord_state(i, j)
                actions = self.world.get_actions(state)
                if actions:
                    v = [self.getQ(state, a) for a in actions]
                    V = np.array(v)
                    A = np.array(actions)
                    R = V.dot(A)
                    X[i, j] = R[0]
                    Y[i, j] = R[1]
                else:
                    X[i, j] = 0
                    Y[i, j] = 0
        plt.gcf().clear()
        current = np.zeros((self.world.board_height, self.world.board_width))
        current[self.world.current_pos] = 3
        plt.imshow(self.world.board + current, interpolation='none')
        #plt.imshow(self.world.board, interpolation='none')
        plt.quiver(Y, -1 * X, color='w')
        plt.title('Episode #'+str(k))
        plt.pause(0.1)
        if save_flag is True:
             plt.savefig('./figs1/'+image_names+string.zfill(str(k), 5)+'.png' , format='png')
             plt.show()