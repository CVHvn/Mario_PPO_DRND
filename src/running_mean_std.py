#https://github.com/jcwleo/random-network-distillation-pytorch/blob/master/utils.py

import torch
import numpy as np

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), device="cpu"):
        self.mean = torch.tensor(np.zeros(shape, 'float32'))#.to(device)
        self.var = torch.tensor(np.ones(shape, 'float32'))#.to(device)
        self.count = epsilon

    def update(self, x):
        x = x.float()
        batch_mean = x.mean(0)
        batch_var = torch.var(x, 0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count