"""
Functions for calculating gradients
"""
import numpy as np
import torch

def gfunc1(G, s_log):
    """
    Gradient function in question 1.1
    :param G: A 1-d array for rewards collected in an episode
    :param s_log: A 1-d array for loglikelihoods collected in an episode
    :return:
    """
    return torch.sum(G) * torch.sum(s_log)

def gfunc2(G, s_log):
    total = 0
    for i in range(s_log.shape[0]):
        total += s_log[i] * torch.sum(G[i:])
    return total

def gfunc3(G, s_log, mu, std):
    assert not np.isnan(mu) and not np.isnan(std)
    total = 0
    for i in range(s_log.shape[0]):
        if std != None and std != 0:
            total += s_log[i] * (torch.sum((G[i:] - mu) / std))
        else:
            total += s_log[i] * (torch.sum(G[i:] - mu))
    return total

def perform(f, *args):
    return f(*args)