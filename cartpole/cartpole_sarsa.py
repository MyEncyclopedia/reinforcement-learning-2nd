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

    def to_bin_idx(self, val: float, lower: float, upper: float, bucket_num: int) -> int:
        percent = (val + abs(lower)) / (upper - lower)
        return min(bucket_num - 1, max(0, int(round((bucket_num - 1) * percent))))

    def discretize(self, obs: np.ndarray) -> State:
        discrete_states = tuple([self.to_bin_idx(obs[d], *self.dims[d]) for d in range(len(obs))])
        return discrete_states

    def choose_action(self, state) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q[state])

    def update_sarsa(self, state: State, action: Action, r, state_next: State, action_next: Action):
        self.q[state][action] += self.lr * (r + self.discount * (self.q[state_next][action_next]) - self.q[state][action])

    def get_epsilon(self, t):
        self.epsilon = max(self.min_epsilon, self.epsilon * 0.99)
        return self.epsilon
        # return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def get_learning_rate(self, t):
        # return 0.05
        # return self.min_lr
        # return max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))
        self.lr = max(self.min_lr, self.lr * 0.99)
        print(self.lr)
        return self.lr

        # return self.lr * 0.95


    def train(self):

        for e in range(self.num_episodes):
            print(e)
            s: State = self.discretize(self.env.reset())

            self.lr = self.get_learning_rate(e)
            self.epsilon = self.get_epsilon(e)
            done = False

            while not done:
                action = self.choose_action(s)
                obs, reward, done, _ = self.env.step(action)
                s_prime: State = self.discretize(obs)
                a_prime = self.choose_action(s_prime)
                self.update_sarsa(s, action, reward, s_prime, a_prime)
                s = s_prime

        print('Finished training!')

    def run(self):
        self.env = gym.wrappers.Monitor(self.env, 'cartpole')
        t = 0
        done = False
        current_state = self.discretize(self.env.reset())
        while not done:
            self.env.render()
            t = t + 1
            action = self.choose_action(current_state)
            obs, reward, done, _ = self.env.step(action)
            new_state = self.discretize(obs)
            current_state = new_state

        return t


if __name__ == "__main__":
    agent = CartPoleAgent()
    agent.train()
    for _ in range(5):
        t = agent.run()
        print("Time", t)
