from World import *
from Q_Learner import *
import random
#from matplotlib import style


random.seed(1)
w = World()
plt.ion()
# style.use('fivethirtyeight')

epsilon_vec = [0, 0.3, 0.6, 0.9]
steps_mean_mat = []
reward_mean_mat = []

world = World()
world.set_board_size(1,20)

N_B = 300 # number of learning batches
N_T = 1   # number of learning episodes per batch
N_E = 300 # number of evaluation episodes per batch
maximal_step = 1000  #  maximal steps in episode

Optimum = Q_Learner(world)
for batch in range(N_B):
    v1 = Optimum.train(N_T, maximal_step, disp_flag=True, print_flag=False, disp_policy_flag=True)
    v2, reward_vec = Optimum.eval(N_E, maximal_step, print_flag=False)
    print (np.mean(v2), np.std(v2))
        #Tan.print_policy(batch)
    steps_mean_mat.append(np.mean(v2))
    reward_mean_mat.append(np.mean(reward_vec))

# w.set_random_pos()
# w.board
# w.get_current_state()
# w.get_actions(w.get_current_state())
#
# for n in range(1000):
#     actions = w.get_actions(w.get_current_state())
#     chosen_action = random.choice(actions)
#     w.update_state(chosen_action)
#     w.display()
