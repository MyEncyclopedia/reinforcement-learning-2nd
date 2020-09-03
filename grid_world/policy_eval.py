from GridWorldEnv import GridWorldEnv
from mdp import MDP, Policy, Value, Action, build_mdp, random_policy
import numpy as np

def policy_evaluate(policy: Policy, env: GridWorldEnv, gamma=1.0, theta=0.0001):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return np.array(V)

if __name__ == "__main__":
    env = GridWorldEnv()
    random_policy = np.ones([env.nS, env.nA]) / env.nA
    V = policy_evaluate(random_policy, env)
    print(V)