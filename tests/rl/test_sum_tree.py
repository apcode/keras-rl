from __future__ import division
import pytest
import numpy as np

from rl.sum_tree import SumTree


def test_sumtree_capacity_and_size():
    t = SumTree(1)
    assert(t.capacity == 1)
    assert(t.size == 1)
    t = SumTree(2)
    assert(t.capacity == 2)
    assert(t.size == 3)
    t = SumTree(3)
    assert(t.capacity == 4)
    assert(t.size == 7)
    t = SumTree(9)
    assert(t.capacity == 16)
    assert(t.size == 31)


def test_sumtree_getset():
    t = SumTree(4)
    assert(t.nodes[0] == 0)
    assert(t[0] == 0)
    t[0] = 0.25
    assert(t.nodes[0] == 0.25)
    assert(t.nodes[1] == 0.25)
    assert(t.nodes[3] == 0.25)
    assert(t[0] == 0.25)
    t[1] = 0.25
    assert(t.nodes[0] == 0.5)
    assert(t.nodes[1] == 0.5)
    assert(t.nodes[4] == 0.25)
    assert(t[0] == 0.25)
    assert(t[1] == 0.25)
    t[2] = 0.25
    assert(t.nodes[0] == 0.75)
    assert(t.nodes[1] == 0.5)
    assert(t.nodes[2] == 0.25)
    assert(t[2] == 0.25)
    t[3] = 0.25
    assert(t.nodes[0] == 1.0)
    assert(t.nodes[1] == 0.5)
    assert(t.nodes[2] == 0.5)
    assert(t[3] == 0.25)
    # Non 1.0 update
    t[3] = 1.25
    assert(t.nodes[0] == 2.0)
    assert(t.nodes[1] == 0.5)
    assert(t.nodes[2] == 1.5)
    assert(t[3] == 1.25)


def test_weighted_node():
    t = SumTree(4)
    for i in range(4):
        t[i] = 1.0
    idx = t._get_weighted_node(0.1)
    assert(idx == 3)
    idx = t._get_weighted_node(0.25)
    assert(idx == 4)
    idx = t._get_weighted_node(0.8)
    assert(idx == 6)
    for i in range(4):
        t[i] = i + 1
    # 1, 2, 3, 4 = 10
    idx = t._get_weighted_node(0.09)
    assert(idx == 3)
    idx = t._get_weighted_node(0.1)
    assert(idx == 4)
    idx = t._get_weighted_node(0.4)
    assert(idx == 5)
    idx = t._get_weighted_node(0.6)
    assert(idx == 6)


if __name__ == '__main__':
    pytest.main([__file__])
