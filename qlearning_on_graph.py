import matplotlib.pyplot as plt
import numpy as np


def check_inputs(costs: np.ndarray, policy: np.ndarray):
    assert costs.ndim == 2
    assert policy.ndim == 1
    assert costs.shape[0] == policy.shape[0]
    assert policy.dtype == int

def apply_td0_algorithm(costs: np.ndarray, policy: np.ndarray, gamma=0.8, alpha=0.2, num_iterations=1000, vertex_for_history: int = 0):
    check_inputs(costs, policy)

    returns = np.zeros(len(costs))
    returns_history = []

    v = 0
    for i in range(num_iterations):
        returns_history.append(returns[vertex_for_history])
        vnext = policy[v]
        reward = costs[v, vnext]
        returns[v] = returns[v] + alpha * (reward + gamma * returns[vnext] - returns[v])
        v = vnext
    return returns, returns_history


def estimate_policy_monte_carlo(costs: np.ndarray, policy: np.ndarray, gamma=0.8, num_iterations=200):
    check_inputs(costs, policy)

    returns = np.zeros(len(costs))
    for vertex in range(len(costs)):
        g = 0.0
        v = vertex
        scale = 1.0
        for i in range(num_iterations):
            vnext = policy[v]
            reward = costs[v, vnext]
            g += scale * reward
            scale *= gamma
            v = vnext
        returns[vertex] = g
    return returns


if __name__ == "__main__":
    costs = np.array([[0, 2, 9], [1, 0, 5], [3, 2, 0]])
    policy1 = np.array([1, 2, 0])
    policy2 = np.array([2, 0, 1])

    returns1_mc = estimate_policy_monte_carlo(costs, policy1)
    returns2_mc = estimate_policy_monte_carlo(costs, policy2)

    print(f"Returns for policy 1 estimated by MonteCarlo: {returns1_mc}")
    print(f"Returns for policy 2 estimated by MonteCarlo: {returns2_mc}")

    returns1_td0, hist1 = apply_td0_algorithm(costs, policy1)
    returns2_td0, hist2 = apply_td0_algorithm(costs, policy2)

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
        returns, hist = apply_td0_algorithm(costs, policy1, alpha=alpha)
        histories.append(hist)

    for alpha, hist in zip(alpha_list, histories):
        plt.plot(hist, label=f"Alpha {alpha}")
    plt.grid()
    plt.legend()
    plt.show()




