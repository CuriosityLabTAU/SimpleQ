import neuronets3 as neuronets
import random
import matplotlib.pyplot as plt
import numpy as np

nInput = 2
nHidden = 5
nOut = 1
eta1 = 0.01
eps1 = 0.01

nn = neuronets.NN(nInput, nHidden, nOut, eta1, eps1)
nn.initialize_weights()


steps = 10000
x = np.zeros((1, 2))
J = np.zeros((1, steps))
log_y = np.zeros((1, steps))
log_d = np.zeros((1, steps))
time_step = np.zeros((1, steps))
x_old = 0
for i in range(0,steps):
    time_step[0, i] = i

    temp = random.uniform(0, 1)
    #if temp>0.5:
    #    x[1,0] = 0.001
    #else:
    #    x[1,0] = -0.001
    #x[0, 0] = x_old
    #d = x[0, 0] + x[1, 0]

    a = random.uniform(0, 0.5)
    b = random.uniform(0, 0.5)
    x[0, 0] = a
    x[0, 1] = b
    d = (x[0,0] + x[0,1])
    J[0, i] = nn.learn(x, d)
    print(x,d, J[0,i])
    #xa, s1, za, s2, y = nn.forProp(x)
    #J[0, i] = nn.backProp(xa, s1, za, s2, y, d)
    #log_y[0, i] = y
    #log_d[0, i] = d
    x_old = d

plt.figure(1)
plt.plot(J[0,:])
#plt.figure(2)
#plt.plot(time_step[0, :], np.abs(log_y[0, :]-log_d[0, :]))
plt.show()

