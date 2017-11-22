import numpy as np
import pandas as pd
import math


def norm(x):
    return math.sqrt(np.sum(x ** 2))


def dist(x, y):
    return norm(x - y)


class GradientDescent:
    def __init__(self, path):
        data = np.array(pd.read_csv(path, sep=' ', header=None))
        self.A = data[1:]
        self.d = data[:1][0]
        self.all_phi = []

    def compute_tau_k(self, phi_k):
        r_k = self.compute_r_k(phi_k)
        return np.dot(r_k, r_k.T) / np.dot(np.dot(self.A, r_k), r_k.T)

    def compute_r_k(self, phi_k):
        return np.dot(self.A, phi_k) - self.d

    def next_phi(self, phi_k):
        tau_k = self.compute_tau_k(phi_k)
        return phi_k - tau_k*(np.dot(self.A, phi_k) - self.d)

    def run(self, eps=0.001, phi_0=None):
        if phi_0 is None:
            phi_0 = [0, 0, 0]

        phi_1 = self.next_phi(phi_0)

        self.all_phi = [phi_0, phi_1]
        iters = 0
        max_iters = 10000
        while iters < max_iters and dist(self.all_phi[-2], self.all_phi[-1]) > eps:
            phi = self.all_phi[-1]
            next_phi = self.next_phi(self.all_phi[-1])
            self.all_phi.append(next_phi)
            print('k=', iters, phi, 'tau=', self.compute_tau_k(phi), 'rk=', self.compute_r_k(phi),
                  'dist=', dist(self.all_phi[-2], self.all_phi[-1]))
            iters += 1

        print('Iter number = ', iters)
        print('Result: ', self.all_phi[-1])
        self.all_phi = list()


task = GradientDescent('res/matrix')
task.run()
