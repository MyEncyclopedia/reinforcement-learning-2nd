from mdp import MDP, Policy, Value, Action, build_mdp, random_policy


def policy_eval(mdp: MDP, policy: Policy) -> Value:
    V = [0] * 16
    gamma = 1.0
    theta = 0.01

    while True:
        delta = 0.0
        for s in range(1, 4*4-1):
            orig_v = V[s]
            new_v = 0
            for action in list(Action):
                p_pi = policy[s][action]
                sum_s_r = sum([p_transition * (r + gamma * V[s_]) for (s_, r, p_transition) in mdp[(s, action)]])
                new_v += p_pi * sum_s_r
            V[s] = new_v
            delta = max(delta, abs(orig_v - new_v))
        print(delta)
        print(V)
        print()
        if delta < theta:
            break
    return V

if __name__ == "__main__":
    mdp = build_mdp()
    V = policy_eval(mdp, random_policy())