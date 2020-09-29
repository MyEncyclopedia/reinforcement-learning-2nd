import random
from collections import defaultdict
from itertools import product
import numpy as np

from gym.envs.toy_text import BlackjackEnv

from blackjack.common import gen_custom_s0_episode_data
from blackjack.plotting import plot_value_function


def mc_control_exploring_starts(env: BlackjackEnv, num_episodes, discount_factor=1.0):
    # states = list(product(range(10, 22), range(1, 11), (True, False)))
    states = list(product(range(10, 22), range(1, 11), (True, False)))
    policy = {s: np.ones(env.action_space.n) * 1.0 / env.action_space.n for s in states}
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for episode_i in range(1, num_episodes + 1):
        # s_0 = random.choice(states)
        s_0 = (random.choice(range(10, 22)), random.choice(range(1, 11)), True)

        player_sum = s_0[0]
        oppo_sum = s_0[1]
        has_usable = s_0[2]

        env.reset()
        env.dealer[0] = oppo_sum
        if has_usable:
            env.player[0] = 1
            env.player[1] = player_sum - 11
        else:
            if player_sum > 11:
                env.player[0] = 10
                env.player[1] = player_sum - 10
            else:
                env.player[0] = 2
                env.player[1] = player_sum - 2

        episode_history = gen_custom_s0_episode_data(policy, env, s_0)

        G = 0
        for t in range(len(episode_history) - 1, -1, -1):
            s, a, r = episode_history[t]
            G = discount_factor * G + r
            if not any(s_a_r[0] == s and s_a_r[1] == a for s_a_r in episode_history[0: t]):
                returns_sum[s, a] += G
                returns_count[s, a] += 1.0
                Q[s][a] = returns_sum[s, a] / returns_count[s, a]
                best_a = np.argmax(Q[s])
                policy[s][best_a] = 1.0
                policy[s][1-best_a] = 0.0

    return Q, policy


if __name__ == "__main__":
    env = BlackjackEnv()
    Q, policy = mc_control_exploring_starts(env, num_episodes=40000)
    print(policy)

    actions = {True: [], False: []}
    for usable in (True, False):
        for player in range(11, 22):
            row = []
            for opponent in range(1, 11):
                s = (player, opponent, usable)
                best_a = np.argmax(policy[s])
                row.append(best_a)
                print(f'{s} {best_a}')
            actions[usable].append(row)


    # V = defaultdict(float)
    # for state, actions in Q.items():
    #     action_value = np.max(actions)
    #     V[state] = action_value
    # plot_value_function(V, title="10,000 Steps")

    import matplotlib.pyplot as plt

    # x, y = np.meshgrid(range(1, 11), range(11, 22))
    #
    # intensity = np.array(actions[True])
    #
    # plt.pcolormesh(x, y, intensity)
    # plt.colorbar() #need a colorbar to show the intensity scale
    # plt.show() #boom
    #
    # intensity = np.array(actions[False])
    #
    # plt.pcolormesh(x, y, intensity)
    # plt.colorbar() #need a colorbar to show the intensity scale
    # plt.show() #boom

    # plot_value_function(V_10k, title="10,000 Steps")

    # V_500k = mc_prediction(sample_policy, env, num_episodes=50000)
    # plotting.plot_value_function(V_500k, title="500,000 Steps")
