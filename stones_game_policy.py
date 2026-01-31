import random
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np


class StonesGame:
    def __init__(self, possible_stones_number):
        self.possible_stones_number = possible_stones_number
        self.dct_returns = defaultdict(float) # return for action-position pair is the (expected) gain
        # that the user obtains if he starts in this position with this action

    def get_best_action(self, stones_number):
        possible_actions = [num for num in self.possible_stones_number if num <= stones_number]
        rews = defaultdict(float)
        for action in possible_actions:
            rews[action] = self.dct_returns[(stones_number, action)]
        key_best = max(rews, key=rews.get)
        return key_best, rews[key_best]

    def get_random_action(self, stones_number):
        possible_actions = [num for num in self.possible_stones_number if num <= stones_number]
        return random.choice(possible_actions)


class StonesGamePolicy(ABC):
    @abstractmethod
    def next_state(self, stones_number, gamer_idx):
        pass


class EpsGreedyStonesGamePolicy(StonesGamePolicy):
    def __init__(self, game: StonesGame, epsilon0: float, epsilon1: float, gamma: float):
        super().__init__()
        self.game = game
        self.epsilons = np.array([epsilon0, epsilon1])
        self.gamma = gamma

    def next_state(self, stones_number, gamer_idx):
        p = np.random.uniform(low=0, high=1)
        if p < self.epsilons[gamer_idx]:
            return stones_number - self.game.get_random_action(stones_number)
        else:
            best_act, best_reward = self.game.get_best_action(stones_number)
            return stones_number - best_act