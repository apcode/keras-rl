"""A sum tree data structure, required for prioritised replay memory.

Used to index into the replay memory and select samples based on their
priority value.

Usage: (see tests/rl/test_sum_tree.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np


class SumTree:
    """A sum tree is a binary tree whose nodes contain the sum of
    both their child nodes. The leaf nodes represent the priorities
    of the samples. Thus the sum tree can be used to select a sample
    based on their priorities, i.e. entries are selected in proportion to
    their priorities.

    The nodes of the tree are stored in a single numpy array.
    [ 0-0, 1-0, 1-1, 2-0, 2-1, 2-2, 2-3 ]
      0    1    2    3    4    5    6
    child index:
      left:  (i + 1) * 2 - 1
      right: (i + 1) * 2
    parent index:
      child: (i - 1) // 2  if i != 0
    """

    def __init__(self, capacity):
        """Create the sum tree with the provided capacity.

        The internal capacity os rounded up to the nearest power of 2.
        So best to create a structure with such a capacity.
        """
        self.capacity = 2 ** math.ceil(math.log(capacity, 2))
        self.size = self.capacity * 2 - 1
        self.nodes = np.zeros(self.size)

    def __getitem__(self, index):
        """Get the priority value at index"""
        return self.nodes[-self.capacity + index]

    def __setitem__(self, index, priority):
        """Set node[index] to priority value

        Update parent sums.
        """
        idx = self.size - self.capacity + index
        diff = priority - self.__getitem__(index)
        self.nodes[idx] = priority
        # update parent nodes
        while True:
            idx = (idx - 1) // 2
            self.nodes[idx] += diff
            if idx == 0:
                break

    def _get_weighted_node(self, weight):
        """Get index given weight
        Separate function for testing.
        """
        w = weight * self.nodes[0]
        node = 0
        while node < self.size - self.capacity:
            left = (node + 1) * 2 - 1
            if w < self.nodes[left]:
                node = left
            else:
                node = left + 1
                w -= self.nodes[left]
        return node

    def sample(self, batch_size):
        """Sample weighted batch based on priorities.
        """
        weights = np.random.random(batch_size)
        batch = [self._get_weighted_node(w) for w in weights]
        return batch
