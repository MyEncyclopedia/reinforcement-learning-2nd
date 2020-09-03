import sys

from gym.envs.toy_text import discrete
import numpy as np

from enum import Enum
from typing import Dict, Tuple, List, Set, io


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

State = int
Reward = float
Prob = float
# MDP = Dict[Tuple[State, Action], List[Tuple[State, Reward, Prob]]]
Policy = Dict[State, Dict[Action, Prob]]
Value = List[float]
StateSet = Set[int]
NonTerminalStateSet = Set[int]

# P[s][a] = [(prob, next_state, reward, is_done), ...]
MDP = Dict[State, Dict[Action, List[Tuple[Prob, State, Reward, bool]]]]

class GridWorldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.

    For example, a 4x4 grid looks as follows:

    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T

    x is your position and T are the two terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    def __init__(self, shape=[4,4]):
        self.shape = shape

        nS = np.prod(shape)
        nA = len(list(Action))

        MAX_R = shape[0]
        MAX_C = shape[1]
        self.grid = np.arange(nS).reshape(shape)

        isd = np.ones(nS) / nS

        # P[s][a] = [(prob, next_state, reward, is_done), ...]
        P: MDP = {}
        action_delta = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}
        for s in range(0, MAX_R * MAX_C):
            P[s] = {a.value : [] for a in list(Action)}
            is_terminal = self.is_terminal(s)
            if is_terminal:
                for a in list(Action):
                    P[s][a.value] = [(1.0, s, 0, True)]
            else:
                r = s // MAX_R
                c = s % MAX_R
                for a in list(Action):
                    neighbor_r = min(MAX_R-1, max(0, r + action_delta[a][0]))
                    neighbor_c = min(MAX_C-1, max(0, c + action_delta[a][1]))
                    s_ = neighbor_r * MAX_R + neighbor_c
                    P[s][a.value] = [(1.0, s_, -1, False)]

        super(GridWorldEnv, self).__init__(nS, nA, P, isd)

    def is_terminal(self, s: State) -> bool:
        return (s == 0) or (s == self.shape[0] * self.shape[1] - 1)

    def _render(self, mode='human', close=False):
        """ Renders the current gridworld layout

         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()

if __name__ == "__main__":
    env = GridWorldEnv()
    print(env)
