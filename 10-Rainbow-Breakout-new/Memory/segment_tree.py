import numpy as np

#求最小值的线段树
class MinSegmentTree:
    def __init__(self, capacity):
        #满二叉树
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.tree = list(np.full(2 * self.capacity - 1, np.inf))

    def query(self, start_idx, end_idx, current_node, first_node, last_node):
        if start_idx == first_node and end_idx == last_node:  # 如果我们位于包含所需内容的节点上。
            return self.tree[current_node]
        mid_node = (first_node + last_node) // 2
        if mid_node >= end_idx:  # 如果查找范围完全在左子树上
            return self.query(start_idx, end_idx, 2 * current_node, first_node, mid_node)
        elif mid_node + 1 <= start_idx:  # 如果查找范围完全在右子树上
            return self.query(start_idx, end_idx, 2 * current_node + 1, mid_node, last_node)
        else:  #如果查找范围部分位于左子树上，部分位于右子树上
            return min(self.query(start_idx, mid_node, 2 * current_node, first_node, mid_node),  # 左子树
                       self.query(mid_node + 1, end_idx, 2 * current_node + 1, mid_node + 1, last_node))  # 右子树

    def min(self, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = self.capacity
        elif end_idx < 0:
            end_idx += self.capacity
        end_idx -= 1
        return self.query(start_idx, end_idx, 1, 0, self.capacity - 1)

    def __setitem__(self, idx, value):
        idx += self.capacity
        self.tree[idx] = value
        # 通过树来传播变化
        idx //= 2
        while idx >= 1:
            self.tree[idx] = min(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self.capacity
        idx += self.capacity
        return self.tree[idx]

#求和的线段树
class SumSegmentTree:
    def __init__(self, capacity):
        assert capacity > 0 and capacity & (capacity - 1) == 0  #满二叉树
        self.capacity = capacity
        self.tree = list(np.full(2 * self.capacity - 1, 0))

    def query(self, start_idx, end_idx, current_node, first_node, last_node):
        if start_idx == first_node and end_idx == last_node:  # 如果我们位于包含所需内容的节点上。
            return self.tree[current_node]
        mid_node = (first_node + last_node) // 2
        if mid_node >= end_idx:  # 如果查找范围完全在左子树上
            return self.query(start_idx, end_idx, 2 * current_node, first_node, mid_node)
        elif mid_node + 1 <= start_idx:  # 如果查找范围完全在右子树上
            return self.query(start_idx, end_idx, 2 * current_node + 1, mid_node, last_node)
        else:  #如果查找范围部分位于左子树上，部分位于右子树上
            return self.query(start_idx, mid_node, 2 * current_node, first_node, mid_node) + \
                   self.query(mid_node + 1, end_idx, 2 * current_node + 1, mid_node + 1,
                              last_node)  # 左 + 右

    def sum(self, start_idx=0, end_idx=None):
        if end_idx is None:
            end_idx = self.capacity
        elif end_idx < 0:
            end_idx += self.capacity
        end_idx -= 1
        return self.query(start_idx, end_idx, 1, 0, self.capacity - 1)

    def find_node(self, prior):
        assert 0 <= prior <= self.sum() + 1e-5

        idx = 1  # 根节点
        while idx < self.capacity:
            if self.tree[2 * idx] > prior:  # 左子树
                idx *= 2
            else:
                prior -= self.tree[2 * idx]
                idx = 2 * idx + 1
        return idx - self.capacity

    def __setitem__(self, idx, value):
        idx += self.capacity
        self.tree[idx] = value
        # 通过树来传播变化
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self.capacity
        idx += self.capacity
        return self.tree[idx]
