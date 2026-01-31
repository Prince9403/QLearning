import typing as t

import matplotlib.pyplot as plt
import numpy as np

from stones_game_policy import StonesGame, StonesGamePolicy, EpsGreedyStonesGamePolicy


def qlearning_for_policy(pol: StonesGamePolicy, num_stones: int, alpha: float, num_games: int):
    gamma = pol.gamma
    game = pol.game

    for i in range(num_games):
        curr_num_stones = num_stones
        gamer_idx = 0
        while curr_num_stones > 0:
            next_num_stones = pol.next_state(curr_num_stones, gamer_idx)
            action = curr_num_stones - next_num_stones

            if next_num_stones > 0:
                rew = 0
            else:
                rew = 1.0

            gamer_idx = 1 - gamer_idx

            if next_num_stones == 0:
                game.dct_returns[(curr_num_stones, action)] = (1 - alpha) * game.dct_returns[(curr_num_stones, action)] + alpha * rew
            else:
                _, best_q = game.get_best_action(next_num_stones)
                # best_q for the adversary is loose for us, so it goes with "minus" sign
                game.dct_returns[(curr_num_stones, action)] += alpha * (rew + gamma * (-best_q) - game.dct_returns[(curr_num_stones, action)])
            curr_num_stones = next_num_stones


if __name__ == "__main__":
    game = StonesGame([1, 2, 3])

    pol = EpsGreedyStonesGamePolicy(game, 0.1, 0.1, 1.0)
    qlearning_for_policy(pol, 20, 0.3, 200000)

    print(pol.game.dct_returns)

    num_stones = 19
    gamer_idx = 0

    print(f"Number of stones; {num_stones}")

    while num_stones > 0:
        best_action, _ = game.get_best_action(num_stones)
        print(f"{num_stones} stones. Player {gamer_idx} took {best_action} stones, remained {num_stones - best_action} stones")
        num_stones = num_stones - best_action
        gamer_idx = 1 - gamer_idx



