import typing as t

import matplotlib.pyplot as plt
import numpy as np

from policy import Graph, GraphPolicy, EpsGreedyGraphPolicy


def qlearning_for_policy(pol: GraphPolicy, alpha: float, num_steps: int, state_action: t.Tuple[int, int]):
    history = []

    gamma = pol.gamma
    graph = pol.graph

    v = 0
    for i in range(num_steps):
        history.append(graph.dct_returns[state_action])
        vnext = pol.next_state(v)
        rew = graph.costs[v, vnext]

        _, max_q = graph.get_best_action(v, gamma)
        graph.dct_returns[(v, vnext)] += alpha * (rew + gamma * max_q - graph.dct_returns[(v, vnext)])
        v = vnext
    return history


if __name__ == "__main__":
    costs = np.array([[0, 2, 9], [1, 0, 5], [3, 2, 0]])

    epsilon = 0.1

    alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    histories = []

    for alpha in alpha_list:

        gamma = 0.9

        graph = Graph(costs)

        pol = EpsGreedyGraphPolicy(graph, epsilon, gamma)

        history = qlearning_for_policy(pol, alpha, num_steps=1500, state_action=(0, 2))
        histories.append(history)

        print(graph.dct_returns)

    for alpha, history in zip(alpha_list, histories):
        plt.plot(history, label=f"Alpha {alpha}")
    plt.legend()
    plt.grid()
    plt.show()

