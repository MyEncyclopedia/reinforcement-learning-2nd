from GridWorldEnv import GridWorldEnv, Policy, Value
import numpy as np

from plot import matplot_bar3d_ex


def policy_evaluate(policy: Policy, env: GridWorldEnv, gamma=1.0, theta=0.0001) -> Value:
    V = np.zeros(env.nS)
    k = 0
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # matplot_bar3d_ex(V, f'{k}')
        k += 1

        if delta < theta:
            break
    return np.array(V)

if __name__ == "__main__":
    env = GridWorldEnv()
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    V = policy_evaluate(random_policy, env)
    print(V.reshape(4, 4))
    # matplot_bar3d_ex(V )

    print(V)