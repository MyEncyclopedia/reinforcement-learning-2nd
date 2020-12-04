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

from .wrappers import wrap_environment
from .model import CNNDQN

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

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


ACTION_SPACE = 'complex'
BATCH_SIZE = 32
BETA_FRAMES = 10000
BETA_START = 0.4
ENVIRONMENT = 'SuperMarioBros-1-1-v0'
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
EPSILON_DECAY = 100000
GAMMA = 0.99
INITIAL_LEARNING = 10000
LEARNING_RATE = 1e-4
MEMORY_CAPACITY = 20000
NUM_EPISODES = 50000
PRETRAINED_MODELS = 'pretrained_models'
TARGET_UPDATE_FREQUENCY = 1000


def update_graph(model, target_model, optimizer, replay_buffer, args, device,
                 episode_idx, beta):
    if len(replay_buffer) > args.initial_learning:
        if not episode_idx % args.target_update_frequency:
            target_model.load_state_dict(model.state_dict())
        optimizer.zero_grad()
        compute_td_loss(model, target_model, replay_buffer, args.gamma, device,
                        args.batch_size, beta)
        optimizer.step()


def exercise_new_model(model, environment, info, action_space):
    # save(model.state_dict(), join(PRETRAINED_MODELS, '%s.dat' % environment))
    print('Testing model...')
    # flag = test(environment, action_space, info.new_best_counter)
    # if flag:
    #     copyfile(join(PRETRAINED_MODELS, '%s.dat' % environment),
    #              'recording/run%s/%s.dat' % (info.new_best_counter,
    #                                          environment))


def complete_episode(model, environment, episode_reward, episode_idx, epsilon, stats, action_space):
    pass
    # new_best = info.update_rewards(episode_reward)
    # if new_best:
    #     print('New best average reward of %s! Saving model'
    #           % round(info.best_average, 3))
    #     exercise_new_model(model, environment, info, action_space)
    # elif stats['flag_get']:
    #     info.update_best_counter()
    #     exercise_new_model(model, environment, info, action_space)
    # print('Episode %s - Reward: %s, Best: %s, Average: %s '
    #       'Epsilon: %s' % (episode,
    #                        round(episode_reward, 3),
    #                        round(info.best_reward, 3),
    #                        round(info.average, 3),
    #                        round(epsilon, 4)))


def run_episode(env, model, target_model, optimizer, replay_buffer, args,
                device, episode_idx):
    episode_reward = 0.0
    state = env.reset()

    while True:
        epsilon = update_epsilon(episode_idx, args)
        if len(replay_buffer) > args.batch_size:
            beta = update_beta(episode_idx, args)
        else:
            beta = args.beta_start
        action = model.act(state, epsilon, device)
        if args.render:
            env.render()
        next_state, reward, done, stats = env.step(action)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        update_graph(model, target_model, optimizer, replay_buffer, args,
                     device, episode_idx, beta)
        if done:
            complete_episode(model, args.environment, episode_reward,
                             episode_idx, epsilon, stats, args.action_space)
            break


def train(env, model, target_model, optimizer, replay_buffer, args, device):
    for episode in range(args.num_episodes):
        run_episode(env, model, target_model, optimizer, replay_buffer, args,
                    device, episode)


def parse_args():
    parser = ArgumentParser(description='')
    parser.add_argument('--batch-size', type=int, help='Specify the batch '
                        'size to use when updating the replay buffer. '
                        'Default: %s' % BATCH_SIZE, default=BATCH_SIZE)
    parser.add_argument('--beta-frames', type=int, help='The number of frames '
                        'to update the beta value before reaching the maximum '
                        'of 1.0. Default: %s' % BETA_FRAMES,
                        default=BETA_FRAMES)
    parser.add_argument('--beta-start', type=float, help='The initial value '
                        'of beta to be used in the prioritized replay. '
                        'Default: %s' % BETA_START, default=BETA_START)
    parser.add_argument('--buffer-capacity', type=int, help='The capacity to '
                        'use in the experience replay buffer. Default: %s'
                        % MEMORY_CAPACITY, default=MEMORY_CAPACITY)
    parser.add_argument('--environment', type=str, help='The OpenAI gym '
                        'environment to use. Default: %s' % ENVIRONMENT,
                        default=ENVIRONMENT)
    parser.add_argument('--epsilon-start', type=float, help='The initial '
                        'value for epsilon to be used in the epsilon-greedy '
                        'algorithm. Default: %s' % EPSILON_START,
                        choices=[Range(0.0, 1.0)], default=EPSILON_START,
                        metavar='EPSILON_START')
    parser.add_argument('--epsilon-final', type=float, help='The final value '
                        'for epislon to be used in the epsilon-greedy '
                        'algorithm. Default: %s' % EPSILON_FINAL,
                        choices=[Range(0.0, 1.0)], default=EPSILON_FINAL,
                        metavar='EPSILON_FINAL')
    parser.add_argument('--epsilon-decay', type=int, help='The decay factor '
                        'for epsilon in the epsilon-greedy algorithm. '
                        'Default: %s' % EPSILON_DECAY, default=EPSILON_DECAY)
    parser.add_argument('--force-cpu', action='store_true', help='By default, '
                        'the program will run on the first supported GPU '
                        'identified by the system, if applicable. If a '
                        'supported GPU is installed, but all computations are '
                        'desired to run on the CPU only, specify this '
                        'parameter to ignore the GPUs. All actions will run '
                        'on the CPU if no supported GPUs are found. Default: '
                        'False')
    parser.add_argument('--gamma', type=float, help='Specify the discount '
                        'factor, gamma, to use in the Q-table formula. '
                        'Default: %s' % GAMMA, choices=[Range(0.0, 1.0)],
                        default=GAMMA, metavar='GAMMA')
    parser.add_argument('--initial-learning', type=int, help='The number of '
                        'iterations to explore prior to updating the model '
                        'and begin the learning process. Default: %s'
                        % INITIAL_LEARNING, default=INITIAL_LEARNING)
    parser.add_argument('--learning-rate', type=int, help='The learning rate '
                        'to use for the optimizer. Default: %s'
                        % LEARNING_RATE, default=LEARNING_RATE)
    parser.add_argument('--num-episodes', type=int, help='The number of '
                        'episodes to run in the given environment. Default: '
                        '%s' % NUM_EPISODES, default=NUM_EPISODES)
    parser.add_argument('--render', action='store_true', help='Specify to '
                        'render a visualization in another window of the '
                        'learning process. Note that a Desktop Environment is '
                        'required for visualization. Rendering scenes will '
                        'lower the learning speed. Default: False')
    parser.add_argument('--target-update-frequency', type=int, help='Specify '
                        'the number of iterations to run prior to updating '
                        'target network with the primary network\'s weights. '
                        'Default: %s' % TARGET_UPDATE_FREQUENCY,
                        default=TARGET_UPDATE_FREQUENCY)
    parser.add_argument('--transfer', action='store_true', help='Transfer '
                        'model weights from a previously-trained model to new '
                        'models for faster learning and improved accuracy. '
                        'Default: False')
    return parser.parse_args()

def compute_td_loss(model, target_net, replay_buffer, gamma, device,
                    batch_size, beta):
    batch = replay_buffer.sample(batch_size, beta)
    state, action, reward, next_state, done, indices, weights = batch

    state = Variable(FloatTensor(np.float32(state))).to(device)
    next_state = Variable(FloatTensor(np.float32(next_state))).to(device)
    action = Variable(LongTensor(action)).to(device)
    reward = Variable(FloatTensor(reward)).to(device)
    done = Variable(FloatTensor(done)).to(device)
    weights = Variable(FloatTensor(weights)).to(device)

    q_values = model(state)
    next_q_values = target_net(next_state)

    q_value = q_values.gather(1, action.unsqueeze(-1)).squeeze(-1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - expected_q_value.detach()).pow(2) * weights
    prios = loss + 1e-5
    loss = loss.mean()
    loss.backward()
    replay_buffer.update_priorities(indices, prios.data.cpu().numpy())


def update_epsilon(episode, args):
    eps_final = args.epsilon_final
    eps_start = args.epsilon_start
    decay = args.epsilon_decay
    epsilon = eps_final + (eps_start - eps_final) * math.exp(-1 * ((episode + 1) / decay))
    return epsilon


def update_beta(episode, args):
    start = args.beta_start
    frames = args.beta_frames
    beta = start + episode * (1.0 - start) / frames
    return min(1.0, beta)


def set_device(force_cpu):
    # force_cpu = True
    device = torch.device('cpu')
    if not force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    print(device)
    return device


def main():
    args = parse_args()
    env = wrap_environment(args.environment, RIGHT_ONLY)
    device = set_device(args.force_cpu)
    model = CNNDQN(env.observation_space.shape, env.action_space.n).to(device)
    target_model = CNNDQN(env.observation_space.shape, env.action_space.n).to(device)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    replay_buffer = ReplayMemory(args.buffer_capacity)
    train(env, model, target_model, optimizer, replay_buffer, args, device)
    env.close()


if __name__ == '__main__':
    main()
