import random
from Memory.segment_tree import MinSegmentTree, SumSegmentTree
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

# 实现Priority replay buffer
class ReplayMemory:
    def __init__(self, capacity, alpha):
        # 经验池大小
        self.capacity = capacity
        # 最大样本权重
        self.max_priority = 1
        # alpha可以调整TD-error的重要性
        self.alpha = alpha
        self.memory = []

        n_nodes = 1
        while n_nodes < self.capacity:
            n_nodes *= 2
        self.sum_tree = SumSegmentTree(n_nodes)
        self.min_tree = MinSegmentTree(n_nodes)
        self.tree_ptr = 0

    #添加样本
    def add(self, *item):
        if len(self.memory) == self.capacity:
            self.memory.pop(self.tree_ptr)

        self.memory.insert(self.tree_ptr, item)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.capacity

        assert len(self.memory) <= self.capacity

    #采样
    def sample(self, batch_size, beta):
        indices = []
        weights = []
        p_total = self.sum_tree.sum(0, len(self))  
        p_min = self.min_tree.min() / p_total
        max_weight = (p_min * len(self)) ** (-beta)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upper_prior = random.uniform(a, b)
            idx = self.sum_tree.find_node(upper_prior)
            indices.append(idx)
            sample_prob = self.sum_tree[idx] / p_total
            weights.append((len(self) * sample_prob) ** -beta)
        weights = np.asarray(weights) / max_weight

        return [self.memory[index] for index in indices], weights, np.asarray(indices)

    # 更新 Priority replay buffer
    def update_priorities(self, indices, priors):
        assert len(indices) == len(priors)
        assert (priors > 0).all()
        assert 0 <= indices.all() < self.capacity

        for idx, prior in zip(indices, priors):
            # 更新对应位置的值的样本的重要性权重
            self.sum_tree[idx] = prior ** self.alpha
            self.min_tree[idx] = prior ** self.alpha
        # 更新最大的权重值
        self.max_priority = max(self.max_priority, max(priors))

    #返回当前经验池的长度
    def __len__(self):
        return len(self.memory)
