import math
import random
from argparse import ArgumentParser
from collections import namedtuple
import numpy as np
import torch
from torch.autograd import Variable
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT

from torch import save, FloatTensor, LongTensor
from torch.optim import Adam

from super_mario.plot_util import plot_rewards
from super_mario.wrappers import wrap_environment
from super_mario.model import CNNDQN

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class Range:
    def __init__(self, start, end):
        self._start = start
        self._end = end

    def __eq__(self, input_num):
        return self._start <= input_num <= self._end

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


ENV_NAME = 'SuperMarioBros-1-1-v0'
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY = 100000
GAMMA = 0.99
INITIAL_LEARNING = 10000
# INITIAL_LEARNING = 10
LEARNING_RATE = 1e-4
MEMORY_CAPACITY = 20000
NUM_EPISODES = 50000
TARGET_UPDATE_FREQUENCY = 1000


def update_graph(model, target_model, optimizer, replay_mem: ReplayMemory, args, device, episode_idx):
    if len(replay_mem) > args.initial_learning:
        if not episode_idx % args.target_update_frequency:
            target_model.load_state_dict(model.state_dict())
        optimizer.zero_grad()
        compute_td_loss(model, target_model, replay_mem, args.gamma, device, args.batch_size)
        optimizer.step()


def exercise_new_model(model, environment, info, action_space):
    # save(model.state_dict(), join(PRETRAINED_MODELS, '%s.dat' % environment))
    print('Testing model...')
    # flag = test(environment, action_space, info.new_best_counter)
    # if flag:
    #     copyfile(join(PRETRAINED_MODELS, '%s.dat' % environment),
    #              'recording/run%s/%s.dat' % (info.new_best_counter,
    #                                          environment))


def train(env, model, target_model, optimizer, replay_mem: ReplayMemory, args, device):
    for episode_idx in range(args.num_episodes):
        episode_reward = 0.0
        state = env.reset()

        while True:
            epsilon = update_epsilon(episode_idx, args)
            action = model.act(state, epsilon, device)
            if args.render:
                env.render()
            next_state, reward, done, stats = env.step(action)
            replay_mem.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            update_graph(model, target_model, optimizer, replay_mem, args, device, episode_idx)
            if done:
                print(f'{episode_idx}: {episode_reward}')
                # plot_rewards(episode_reward)
                break


def parse_args():
    parser = ArgumentParser(description='')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--buffer-capacity', type=int, default=MEMORY_CAPACITY)
    parser.add_argument('--env-name', type=str, default=ENV_NAME)
    parser.add_argument('--epsilon-start', type=float, choices=[Range(0.0, 1.0)], default=EPSILON_START)
    parser.add_argument('--epsilon-final', type=float, choices=[Range(0.0, 1.0)], default=EPSILON_FINAL)
    parser.add_argument('--epsilon-decay', type=int, default=EPSILON_DECAY)
    parser.add_argument('--gamma', type=float, choices=[Range(0.0, 1.0)], default=GAMMA)
    parser.add_argument('--initial-learning', type=int, default=INITIAL_LEARNING)
    parser.add_argument('--learning-rate', type=int, default=LEARNING_RATE)
    parser.add_argument('--num-episodes', type=int, default=NUM_EPISODES)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--target-update-frequency', type=int, default=TARGET_UPDATE_FREQUENCY)

    return parser.parse_args()

def compute_td_loss(model, target_net, replay_mem: ReplayMemory, gamma, device, batch_size):
    transitions = replay_mem.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state = Variable(FloatTensor(np.float32(batch.state))).to(device)
    action = Variable(LongTensor(batch.action)).to(device)
    reward = Variable(FloatTensor(batch.reward)).to(device)
    next_state = Variable(FloatTensor(np.float32(batch.next_state))).to(device)
    done = Variable(FloatTensor(batch.done)).to(device)
    # weights = Variable(FloatTensor(weights)).to(device)

    q_values = model(state)
    next_q_values = target_net(next_state)

    q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    # loss = (q_value - expected_q_value.detach()).pow(2) * weights
    loss = (q_value - expected_q_value.detach()).pow(2)
    prios = loss + 1e-5
    loss = loss.mean()
    loss.backward()


def update_epsilon(episode, args):
    eps_final = args.epsilon_final
    eps_start = args.epsilon_start
    decay = args.epsilon_decay
    epsilon = eps_final + (eps_start - eps_final) * math.exp(-1 * ((episode + 1) / decay))
    return epsilon


def main():
    args = parse_args()
    env = wrap_environment(args.env_name, RIGHT_ONLY)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CNNDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_model = CNNDQN(env.observation_space.shape, env.action_space.n).to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    replay_mem = ReplayMemory(args.buffer_capacity)
    train(env, model, target_model, optimizer, replay_mem, args, device)
    env.close()


if __name__ == '__main__':
    main()
