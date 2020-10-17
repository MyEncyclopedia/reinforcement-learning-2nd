from typing import Tuple

import gym
import numpy as np
import math

State = Tuple[int, int, int, int]
Action = int

class CartPoleAgent():
    def __init__(self, buckets=(1, 2, 6, 12), num_episodes=8000, min_lr=0.1, min_epsilon=0.1, discount=0.98, decay=25):
        self.buckets = buckets
        self.num_episodes = num_episodes
        self.epsilon = 1.0
        self.lr = 1.0
        self.min_lr = min_lr
        self.min_epsilon = min_epsilon
        self.discount = discount
        self.decay = decay

        self.env = gym.make('CartPole-v0')

        # [position, velocity, angle, angular velocity]
        env = self.env
        self.dims = [(env.observation_space.low[0], env.observation_space.high[0], 1),
                     (-0.5, 0.5, 1),
                     (env.observation_space.low[2], env.observation_space.high[2], 6),
                     (-math.radians(50) / 1., math.radians(50) / 1., 12)]
        self.q = np.zeros(self.buckets + (self.env.action_space.n,))

        self.pi = np.zeros_like(self.q)
        for i in range(self.pi.shape[0]):
            for a in range(self.env.action_space.n):
                self.pi[i, a] = 1 / self.env.action_space.n
        # print(self.pi)
        print('done')

    def to_bin_idx(self, val: float, lower: float, upper: float, bucket_num: int) -> int:
        percent = (val + abs(lower)) / (upper - lower)
        return min(bucket_num - 1, max(0, int(round((bucket_num - 1) * percent))))

    def discretize(self, obs: np.ndarray) -> State:
        discrete_states = tuple([self.to_bin_idx(obs[d], *self.dims[d]) for d in range(len(obs))])
        return discrete_states

    def choose_action(self, state) -> Action:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q[state])

    def update_sarsa(self, s: State, a: Action, r, s_next: State, a_next: Action):
        self.q[s][a] += self.lr * (r + self.discount * (self.q[s_next][a_next]) - self.q[s][a])

    def update_q(self, s, a, r, s_next, a_next):
        self.q[s][a] += self.lr * (r + self.discount * np.max(self.q[s_next]) - self.q[s][a])

    def update_exp(self, s, a, r, s_next, a_next):
        # print(np.dot(self.pi[s_next, :], self.q[s_next, :]))
        # print(np.dot(self.pi[s_next], self.q[s_next]))
        self.q[s][a] = self.q[s][a] + self.lr * (r + self.discount * np.dot(self.pi[s_next], self.q[s_next]) - self.q[s][a])
        best_a = np.random.choice(np.where(self.q[s] == max(self.q[s]))[0])
        n_actions = self.env.action_space.n
        for i in range(n_actions):
            if i == best_a:
                self.pi[s][i] = 1 - (n_actions - 1) * (self.epsilon / n_actions)
            else:
                self.pi[s][i] = self.epsilon / n_actions

    def adjust_epsilon(self, t) -> float:
        self.epsilon = max(self.min_epsilon, self.epsilon * 0.99)
        return self.epsilon

    def adjust_learning_rate(self, t) -> float:
        self.lr = max(self.min_lr, self.lr * 0.99)
        return self.lr

    def train(self):
        for e in range(self.num_episodes):
            print(e)
            s: State = self.discretize(self.env.reset())

            self.adjust_learning_rate(e)
            self.adjust_epsilon(e)
            done = False

            while not done:
                action: Action = self.choose_action(s)
                obs, reward, done, _ = self.env.step(action)
                s_next: State = self.discretize(obs)
                a_next = self.choose_action(s_next)
                # self.update_sarsa(s, action, reward, s_next, a_next)
                # self.update_q(s, action, reward, s_next, a_next)
                self.update_exp(s, action, reward, s_next, a_next)
                s = s_next

        print('Finished training!')

    def test(self):
        self.env = gym.wrappers.Monitor(self.env, 'cartpole')
        t = 0
        done = False
        s: State = self.discretize(self.env.reset())
        while not done:
            self.env.render()
            t += 1
            action: Action = self.choose_action(s)
            obs, reward, done, _ = self.env.step(action)
            s_next = self.discretize(obs)
            s = s_next

        return t


if __name__ == "__main__":
    agent = CartPoleAgent()
    agent.train()
    for _ in range(5):
        t = agent.test()
        print("Time", t)
