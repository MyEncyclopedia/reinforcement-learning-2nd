from mdp import MDP, Policy


def policy_improvement(mdp: MDP, policy: Policy):
    policy_stable = True

    for s in range(1, 4 * 4 - 1):