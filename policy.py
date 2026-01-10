from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class Graph:
    def __init__(self, costs: np.ndarray):
        self.costs = costs
        self.n = len(self.costs)
        self.dct_returns = defaultdict(float)

    def get_best_action(self, v, gamma):
        rews = np.zeros(self.n)
        for vnext in range(self.n):
            rews[vnext] = self.dct_returns[(v, vnext)]
        return np.argmax(rews)

    def get_random_action(self, v):
        return np.random.randint(0, self.n)


class GraphPolicy(ABC):
    @abstractmethod
    def next_state(self, state):
        pass

class EpsGreedyGraphPolicy(GraphPolicy):
    def __init__(self, graph: Graph, epsilon: float, gamma: float):
        super().__init__()
        self.graph = graph
        self.epsilon = epsilon
        self.gamma = gamma

    def next_state(self, vertex):
        p = np.random.uniform(low=0, high=1)
        if p < self.epsilon:
            return self.graph.get_random_action(vertex)
        else:
            return self.graph.get_best_action(vertex, self.gamma)
