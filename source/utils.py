import numpy as np
import torch
from torch.distributions.categorical import Categorical
import random
import torch.nn as nn
import logging
from collections import namedtuple
Experience = namedtuple('Experience',
                        ('states', 'actions', 'next_states', 'rewards'))

# Polyak Averaging
def soft_update(target, source, t=0.005):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)

# Simply copy weights to target network
def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)

# Replay Buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# select actions
def select_actions(pi):
    print(pi)
    actions = s
    print(actions)
    # return actions
    return actions.detach().cpu().numpy().squeeze()



# evaluate actions
def evaluate_actions(pi, actions):
    cate_dist = Categorical(pi)
    log_prob = cate_dist.log_prob(actions).unsqueeze(-1)
    entropy = cate_dist.entropy().mean()
    return log_prob, entropy

# configure the logger
def config_logger(log_dir):
    logger = logging.getLogger()
    # we don't do the debug...
    logger.setLevel('INFO')
    basic_format = '%(message)s'
    formatter = logging.Formatter(basic_format)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    # set the log file handler
    fhlr = logging.FileHandler(log_dir)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    return logger
