import math

import torch
from torch import save, FloatTensor, LongTensor
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
from super_mario.common import ReplayMemory, Transition
from super_mario.dqn_agent import DQNAgent
from super_mario.model import DQNModel


class DoubleDQNAgent(DQNAgent):
    def __init__(self, env, buffer_capacity, epsilon_start, epsilon_final, epsilon_decay, lr, initial_learning, gamma, target_update_frequency):
        super().__init__(env, buffer_capacity, epsilon_start, epsilon_final, epsilon_decay, lr, initial_learning, gamma, target_update_frequency)

    def update_epsilon(self, episode_idx):
        self.epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * math.exp(-1 * ((episode_idx + 1) / self.epsilon_decay))
        return self.epsilon

    def act(self, state, episode_idx):
        self.update_epsilon(episode_idx)
        action = self.model.act(state, self.epsilon, self.device)
        return action

    def process(self, episode_idx, state, action, reward, next_state, done):
        self.replay_mem.push(state, action, reward, next_state, done)
        self.train(episode_idx)

    def train(self, episode_idx):
        if len(self.replay_mem) > self.initial_learning:
            if not episode_idx % self.target_update_frequency:
                self.target_model.load_state_dict(self.model.state_dict())
            self.optimizer.zero_grad()
            self.td_loss_backprop()
            self.optimizer.step()

    def td_loss_backprop(self):
        transitions = self.replay_mem.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state = Variable(FloatTensor(np.float32(batch.state))).to(self.device)
        action = Variable(LongTensor(batch.action)).to(self.device)
        reward = Variable(FloatTensor(batch.reward)).to(self.device)
        next_state = Variable(FloatTensor(np.float32(batch.next_state))).to(self.device)
        done = Variable(FloatTensor(batch.done)).to(self.device)
        # weights = Variable(FloatTensor(weights)).to(device)

        q_values = self.model(state)
        next_q_values = self.target_net(next_state)

        q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        # loss = (q_value - expected_q_value.detach()).pow(2) * weights
        loss = (q_value - expected_q_value.detach()).pow(2)
        prios = loss + 1e-5
        loss = loss.mean()
        loss.backward()

