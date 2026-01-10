import matplotlib.pyplot as plt
import numpy as np

from policy import Graph, GraphPolicy, DeterministicGraphPolicy


def apply_td0_algorithm(pol: GraphPolicy, gamma=0.9, alpha=0.2, num_iterations=1000, vertex_for_history: int = 0):
    returns_history = []

    graph = pol.graph

    v = 0
    for i in range(num_iterations):
        returns_history.append(graph.returns[vertex_for_history])
        vnext = pol.next_state(v)
        reward = graph.costs[v, vnext]
        graph.returns[v] += alpha * (reward + gamma * graph.returns[vnext] - graph.returns[v])
        v = vnext
    return graph.returns, returns_history


def estimate_policy_monte_carlo(pol: DeterministicGraphPolicy, gamma=0.9, num_iterations=200):

    graph = pol.graph

    for vertex in range(len(costs)):
        g = 0.0
        v = vertex
        scale = 1.0 # discount factor
        for i in range(num_iterations):
            vnext = pol.next_state(v)
            reward = graph.costs[v, vnext]
            g += scale * reward
            scale *= gamma
            v = vnext
        graph.returns[vertex] = g
    return graph.returns


def estimate_graph_policy_monte_carlo(costs: np.ndarray, next_vertices: np.ndarray):
    graph = Graph(costs)
    policy1 = DeterministicGraphPolicy(graph, next_vertices=next_vertices)
    returns1_mc = estimate_policy_monte_carlo(policy1)
    return returns1_mc


def estimate_graph_policy_td0(costs: np.ndarray, next_vertices: np.ndarray, alpha: float):
    graph = Graph(costs)
    policy1 = DeterministicGraphPolicy(graph, next_vertices=next_vertices)
    returns1_mc, returns1_history = apply_td0_algorithm(policy1, alpha=alpha)
    return returns1_mc, returns1_history


if __name__ == "__main__":
    costs = np.array([[0, 2, 9], [1, 0, 5], [3, 2, 0]])

    next_vertices_1 = np.array([1, 2, 0])
    next_vertices_2 = np.array([2, 0, 1])

    returns1_mc = estimate_graph_policy_monte_carlo(costs, next_vertices_1)
    returns2_mc = estimate_graph_policy_monte_carlo(costs, next_vertices_2)

    print(f"\Returns for policy 1 estimated by MonteCarlo: {returns1_mc}")
    print(f"Returns for policy 2 estimated by MonteCarlo: {returns2_mc}")

    returns1_td0, hist1 = estimate_graph_policy_td0(costs, next_vertices_1, alpha=0.4)
    returns2_td0, hist2 = estimate_graph_policy_td0(costs, next_vertices_2, alpha=0.4)

    print(f"Returns for policy 1 estimated by TD(0): {returns1_td0}")
    print(f"Returns for policy 2 estimated by TD(0): {returns2_td0}")

    plt.plot(hist1, label="History (policy 1)")
    plt.plot(hist2, label="History (policy 2)")
    plt.grid()
    plt.legend()
    plt.show()

    alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    histories = []

    for alpha in alpha_list:
        returns, hist = estimate_graph_policy_td0(costs, next_vertices_1, alpha=alpha)
        histories.append(hist)

    for alpha, hist in zip(alpha_list, histories):
        plt.plot(hist, label=f"Alpha {alpha}")
    plt.grid()
    plt.legend()
    plt.show()




