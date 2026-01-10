import typing as t

import matplotlib.pyplot as plt
import numpy as np

from policy import Graph, GraphPolicy, EpsGreedyGraphPolicy


def sarsa_for_policy(pol: GraphPolicy, alpha: float, num_steps: int, state_action: t.Tuple[int, int]):
    history = []

    gamma = pol.gamma
    graph = pol.graph

    v = 0
    for i in range(num_steps):
        history.append(graph.dct_returns[state_action])
        vnext = pol.next_state(v)
        rew = graph.costs[v, vnext]

        vnext2 = pol.next_state(v)
        graph.dct_returns[(v, vnext)] += alpha * (rew + gamma * graph.dct_returns[(vnext, vnext2)] - graph.dct_returns[(v, vnext)])
        v = vnext
    return history


if __name__ == "__main__":
    costs = np.array([[0, 2, 9], [1, 0, 5], [3, 2, 0]])

    epsilon = 0.1
    alpha = 0.6
    gamma = 0.9

    graph = Graph(costs)

    pol = EpsGreedyGraphPolicy(graph, epsilon, gamma)

    history = sarsa_for_policy(pol, alpha, num_steps=10000, state_action=(0, 2))

    print(graph.dct_returns)

    plt.plot(history)
    plt.grid()
    plt.show()

