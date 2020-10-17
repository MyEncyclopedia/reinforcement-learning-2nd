import abc
from typing import Tuple

import gym
import numpy as np
import math

State = Tuple[int, int, int, int]
Action = int


class CartPoleAbstractAgent(metaclass=abc.ABCMeta):
    def __init__(self, buckets=(1, 2, 6, 12), discount=0.98, lr_min=0.1, epsilon_min=0.1):
        self.epsilon = 1.0
        self.lr = 1.0
        self.discount = discount
        self.lr_decay = 0.99
        self.epsilon_decay = 0.99
        self.lr_min = lr_min
        self.epsilon_min = epsilon_min

        self.env = gym.make('CartPole-v0')

        # [position, velocity, angle, angular velocity]
        env = self.env
        self.dims_config = [(env.observation_space.low[0], env.observation_space.high[0], 1),
                            (-0.5, 0.5, 1),
                            (env.observation_space.low[2], env.observation_space.high[2], 6),
                            (-math.radians(50) / 1., math.radians(50) / 1., 12)]
        self.q = np.zeros(buckets + (self.env.action_space.n,))
        self.pi = np.zeros_like(self.q)
        self.pi[:] = 1.0 / env.action_space.n

    def to_bin_idx(self, val: float, lower: float, upper: float, bucket_num: int) -> int:
        percent = (val + abs(lower)) / (upper - lower)
        return min(bucket_num - 1, max(0, int(round((bucket_num - 1) * percent))))

    def discretize(self, obs: np.ndarray) -> State:
        discrete_states = tuple([self.to_bin_idx(obs[d], *self.dims_config[d]) for d in range(len(obs))])
        return discrete_states

    def choose_action(self, state) -> Action:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q[state])

    @abc.abstractmethod
    def update_q(self, s: State, a: Action, r, s_next: State, a_next: Action):
        pass

    def adjust_epsilon(self, ep: int) -> float:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon
        # return max(self.min_epsilon, min(1., 1. - math.log10((t + 1) / self.decay)))

    def adjust_learning_rate(self, ep: int) -> float:
        self.lr = max(self.lr_min, self.lr * self.lr_decay)
        return self.lr
        # ret = max(self.min_lr, min(1., 1. - math.log10((t + 1) / self.decay)))

    def train(self, num_episodes=2000):
        for e in range(num_episodes):
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
                self.update_q(s, action, reward, s_next, a_next)
                s = s_next

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


class SarsaAgent(CartPoleAbstractAgent):

    def update_q(self, s: State, a: Action, r, s_next: State, a_next: Action):
        self.q[s][a] += self.lr * (r + self.discount * (self.q[s_next][a_next]) - self.q[s][a])


class QLearningAgent(CartPoleAbstractAgent):

    def update_q(self, s: State, a: Action, r, s_next: State, a_next: Action):
        self.q[s][a] += self.lr * (r + self.discount * np.max(self.q[s_next]) - self.q[s][a])


class ExpectedSarsaAgent(CartPoleAbstractAgent):

    def update_q(self, s: State, a: Action, r, s_next: State, a_next: Action):
        self.q[s][a] = self.q[s][a] + self.lr * (r + self.discount * np.dot(self.pi[s_next], self.q[s_next]) - self.q[s][a])
        # update pi[s]
        best_a = np.random.choice(np.where(self.q[s] == max(self.q[s]))[0])
        n_actions = self.env.action_space.n
        self.pi[s][:] = self.epsilon / n_actions
        self.pi[s][best_a] = 1 - (n_actions - 1) * (self.epsilon / n_actions)


if __name__ == "__main__":
    # agent = SarsaAgent()
    agent = QLearningAgent()
    # agent = ExpectedSarsaAgent()
    agent.train(num_episodes=1000)
    for _ in range(3):
        t = agent.test()
        print("Time", t)
