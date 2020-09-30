from __future__ import annotations
from typing import Tuple, Dict, Callable, List
import numpy as np
from gym.envs.toy_text import BlackjackEnv

State = Tuple[int, int, bool]
Action = bool
Reward = float
Actions = np.ndarray
StateValue = Dict[State, float]
ActionValue = Dict[State, np.ndarray]
Policy = Dict[State, Actions]
DeterministicPolicy = Callable[[State], Action]


def gen_episode_data(policy: DeterministicPolicy, env: BlackjackEnv) -> List[Tuple[State, Action, Reward]]:
    episode_history = []
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        next_state, reward, done, _ = env.step(action)
        episode_history.append((state, action, reward))
        state = next_state
    return episode_history


def gen_stochastic_episode(policy: Policy, env: BlackjackEnv) -> List[Tuple[State, Action, Reward]]:
    episode_history = []
    state = env.reset()
    done = False
    while not done:
        A: ActionValue = policy[state]
        action = np.random.choice([0, 1], p=A/sum(A))
        next_state, reward, done, _ = env.step(action)
        episode_history.append((state, action, reward))
        state = next_state
    return episode_history


def gen_custom_s0_stochastic_episode(policy: Policy, env: BlackjackEnv, initial_state: State) \
        -> List[Tuple[State, Action, Reward]]:
    episode_history = []
    state = initial_state
    done = False
    while not done:
        A: ActionValue = policy[state]
        action = np.random.choice([0, 1], p=A/sum(A))
        next_state, reward, done, _ = env.step(action)
        episode_history.append((state, action, reward))
        state = next_state
    return episode_history

def fixed_policy(observation):
    """
    sticks if the player score is >= 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1