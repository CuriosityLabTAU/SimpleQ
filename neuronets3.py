import numpy as np
import matplotlib.pyplot as plt


class NN:

    def __init__(self, nInput, nHidden, nOutput, eta=0.1, eps=0.1, pruning_rate=0.00, pruning_thresh=0.0001, activation=2):
        self.nInput = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        self.eta = eta
        self.eps = eps
        self.pruning_rate = pruning_rate
        self.pruning_thresh = pruning_thresh
        self.activation = activation
        np.random.seed(1)
        #self.initialize_weights()

    def sig(self, z):
        if self.activation == 1:
            h = np.tanh(z)
        elif self.activation == 2:
            h = 1/(1 + np.exp(-z))
        elif self.activation == 3:
            zero = np.zeros((z.shape))
            h = np.maximum(zero, z)
        return h

    def sigtag(self, z):
        if self.activation == 1:
            stag = 1-np.multiply(self.sig(z),self.sig(z))
        elif self.activation == 2:
            stag = self.sig(z)*(1 - self.sig(z))
        elif self.activation == 3:
            stag = np.zeros(z.shape)
            for i in range(0, z.shape[0]):
                if z[i] > 0:
                    stag[i] = 1
                else:
                    stag[i] = 0
        return stag

    def initialize_weights(self, eps_in=None):
        n = self.nInput
        p = self.nHidden
        m = self.nOutput
        if eps_in is not None:
            eps = eps_in
        else:
            eps = self.eps
        self.Wa1 = np.random.rand(p,n+1)*2*eps-eps
        self.Wa2 = np.random.rand(m,p+1)*2*eps-eps

    def forProp(self, x):
        xa = np.insert(x, 0, 1)
        s1 = np.dot(self.Wa1, xa)
        z = self.sig(s1)
        za = np.insert(z, 0, -1)
        s2 = np.dot(self.Wa2, za)
        y = self.sig(s2)
        return xa, s1, za, s2, y

    def cost(self, d, y):
        e = d-y
        J = (0.5)*np.dot(e, e)
        return J

    def backProp(self, xa, s1, za, s2, y, d):
        e2 = d-y
        sigtag2 = self.sigtag(s2)
        d2 = np.multiply(e2, sigtag2)
        D2 = np.outer(-d2, za.T)
        sigtag1 = self.sigtag(s1)
        p = self.Wa2.shape[1]-1
        W2 = self.Wa2[:, 1:p+1]
        e1 = np.dot(W2.T, d2)
        d1 = e1*sigtag1
        D1 = np.outer(-d1, xa.T)
        self.Wa2 -= self.eta * D2# + self.pruning_rate * np.sign(self.Wa2)
        self.Wa1 -= self.eta * D1# + self.pruning_rate * np.sign(self.Wa1)

        return self.cost(d, y)

    def batch_learn(self, x_batch, d_batch):
        D1 = 0
        D2 = 0
        J = 0
        batch_size = x_batch.shape[0]
        eta_batch = self.eta / batch_size
        for i in range(0, batch_size):
            x = x_batch[i, :]
            d = d_batch[i, :]
            xa, s1, za, s2, y = self.forProp(x)
            e2 = d-y
            sigtag2 = self.sigtag(s2)
            d2 = np.multiply(e2, sigtag2)
            D2 += np.outer(-d2, za.T)
            sigtag1 = self.sigtag(s1)
            p = self.Wa2.shape[1]-1
            W2 = self.Wa2[:, 1:p+1]
            e1 = np.dot(W2.T, d2)
            d1 = e1*sigtag1
            D1 += np.outer(-d1, xa.T)
            J += self.cost(d, y) / batch_size
        self.Wa2 -= eta_batch * D2 + self.pruning_rate * np.sign(self.Wa2)
        self.Wa1 -= eta_batch * D1 + self.pruning_rate * np.sign(self.Wa1)
        return J

    def removeNode(self):

        if self.nHidden > 0:
            abs_Wa1 = np.absolute(self.Wa1)
            abs_Wa2 = np.absolute(self.Wa2)

            weight_sum1 = np.sum(abs_Wa1, axis=1)
            weight_sum2 = np.sum(abs_Wa2, axis=0)
            weight_sum2 = weight_sum2[1:]

            total_sum = weight_sum1 + weight_sum2

            input_weight_sum = np.sum(abs_Wa1, axis=0)

            prune_index = 1000

            for i in range(0, self.nHidden):
                if total_sum[i] < self.pruning_thresh:
                    prune_index = i
                    if self.nHidden >= 1:
                        self.nHidden -= 1
                    break

            if prune_index < 999:
                self.Wa1 = np.delete(self.Wa1, prune_index, 0)
                self.Wa2 = np.delete(self.Wa2, prune_index + 1, 1)

    def learn(self, x, d):
        xa, s1, za, s2, y = self.forProp(x)
        J = self.backProp(xa, s1, za, s2, y, d)
        return J








