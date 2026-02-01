import datetime
from abc import ABC, abstractmethod
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


class State:
    def __init__(self, angle_wolf, angle_rabbit):
        self.angle_wolf = angle_wolf
        self.angle_rabbit = angle_rabbit

    def get_possible_actions(self):
        return [RabbitAction(np.pi / 180), RabbitAction(-np.pi / 180)]

    def to_torch_tensor(self):
        positions_np = np.array([np.cos(self.angle_wolf), np.sin(self.angle_wolf), np.cos(self.angle_rabbit),
                                 np.sin(self.angle_rabbit)])
        return torch.from_numpy(positions_np).float()

    def get_distance(self) -> float:
        x_rabbit = np.cos(self.angle_rabbit)
        y_rabbit = np.sin(self.angle_rabbit)
        x_wolf = np.cos(self.angle_wolf)
        y_wolf = np.sin(self.angle_wolf)
        dst2 = (x_rabbit - x_wolf) ** 2 + (y_rabbit - y_wolf) ** 2
        return np.sqrt(dst2)


class RabbitAction:
    def __init__(self, angle: float):
        self.angle = angle

    def to_torch_tensor(self):
        return torch.from_numpy(np.array([self.angle])).float()


class GameOnCircle:
    def __init__(self, angle_wolf, angle_rabbit, distance_threshold, penalty):
        assert distance_threshold > 0
        assert penalty < 0
        self.state = State(angle_wolf, angle_rabbit)
        self.distance_threshold = distance_threshold
        self.distance_threshold2 = self.distance_threshold ** 2
        self.penalty = penalty

    def try_apply_action(self, state: State, action: RabbitAction):
        curr_angle_wolf = state.angle_wolf
        curr_angle_rabbit = state.angle_rabbit
        new_angle_rabbit = (curr_angle_rabbit + action.angle) % (2 * np.pi)
        d_angle_wolf = np.random.choice([np.pi / 180, -np.pi / 180])
        new_angle_wolf = (curr_angle_wolf + d_angle_wolf) % (2 * np.pi)
        state1 = State(new_angle_wolf, new_angle_rabbit)
        return state1

    def apply_action(self, action):
        self.state = self.try_apply_action(self.state, action)

    def get_reward(self, state: State, action: RabbitAction):
        state1 = self.try_apply_action(state, action)
        x_rabbit = np.cos(state1.angle_rabbit)
        y_rabbit = np.sin(state1.angle_rabbit)
        x_wolf = np.cos(state1.angle_wolf)
        y_wolf = np.sin(state1.angle_wolf)
        dst2 = (x_rabbit - x_wolf) ** 2 + (y_rabbit - y_wolf) ** 2
        if dst2 < self.distance_threshold2:
            return self.penalty
        return 0.0


class RabbitNNPolicy(ABC):
    def __init__(self, game_on_circle: GameOnCircle, rabbit_nn: torch.nn.Module):
        self.game_on_circle = game_on_circle
        self.rabbit_nn = rabbit_nn
        self.rabbit_nn_target = deepcopy(rabbit_nn)
        self.rabbit_nn.train()
        self.rabbit_nn_target.eval()

    def evaluate_action(self, state, action, network):
        with torch.no_grad():
            input_for_nn = torch.cat([state.to_torch_tensor(), action.to_torch_tensor()])
            q = network(input_for_nn).item()
        return q

    def get_best_action(self, curr_state, network):
        # find action associated with the largest q value
        possible_actions = curr_state.get_possible_actions()
        qvals = []
        for action in possible_actions:
            q = self.evaluate_action(curr_state, action, network)
            qvals.append(q)
        idx = np.argmax(qvals)
        action = possible_actions[idx]
        return action, qvals[idx]

    @abstractmethod
    def select_action(self):
        pass


class EpsGreedyRabbitNNPolicy(RabbitNNPolicy):
    def __init__(self, game_on_circle: GameOnCircle, rabbit_nn: torch.nn.Module, epsilon: float):
        super().__init__(game_on_circle, rabbit_nn)
        self.epsilon = epsilon

    def select_action(self):
        curr_state = self.game_on_circle.state

        possible_actions = curr_state.get_possible_actions()

        p = np.random.uniform(0, 1)
        if p < self.epsilon:
            action = np.random.choice(possible_actions)
        else:
            action, _ = self.get_best_action(curr_state, rabbit_nn)
        return action


def qlearning_for_nn_policy(pol: RabbitNNPolicy, gamma: float, aplha: float, num_steps: int, nn_copy_freq: int):
    optimizer = torch.optim.Adam(pol.rabbit_nn.parameters(), lr=aplha)

    game = pol.game_on_circle

    history_states = []
    history_q = []
    for i in tqdm(range(num_steps)):
        action = pol.select_action()
        reward = game.get_reward(game.state, action)

        next_state = game.try_apply_action(game.state, action)

        input_for_nn = torch.cat([game.state.to_torch_tensor(), action.to_torch_tensor()])
        q = pol.rabbit_nn(input_for_nn)

        history_q.append(q.item())

        with torch.no_grad():
            _, max_q = pol.get_best_action(next_state, pol.rabbit_nn_target)

        loss = (reward + gamma * max_q - q) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i % nn_copy_freq == 0) or (i == num_steps - 1): # copy weights to target neural network
            pol.rabbit_nn_target = deepcopy(pol.rabbit_nn)
            pol.rabbit_nn.train()
            pol.rabbit_nn_target.eval()

        game.apply_action(action)
        history_states.append(game.state)

    return pol.rabbit_nn, history_states, history_q


if __name__ == "__main__":
    game = GameOnCircle(0.0, np.pi / 2, 0.05, -1.0)

    hidden_dim = 64

    rabbit_nn = torch.nn.Sequential(
        torch.nn.Linear(5, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1)
    )

    eps_greedy_pol = EpsGreedyRabbitNNPolicy(game, rabbit_nn, 0.1)

    print(f"{datetime.datetime.now()} Started learning")

    rabbit_nn, history_states, history_q = qlearning_for_nn_policy(eps_greedy_pol, gamma=0.95, aplha=0.0001, num_steps=100000, nn_copy_freq=200)

    print(f"{datetime.datetime.now()} Ended learning")

    distances = [state.get_distance() for state in history_states]

    plt.subplot(1, 2, 1)
    plt.plot(distances)
    plt.grid()
    plt.title("Distance between wolf and rabbit")
    plt.subplot(1, 2, 2)
    plt.plot(history_q)
    plt.title("QValues")
    plt.grid()
    plt.show()
