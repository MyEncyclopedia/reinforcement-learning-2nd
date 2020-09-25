from __future__ import annotations
from typing import Tuple, Dict, Callable
import numpy as np


State: Tuple[int, bool, int]
Action: bool
Reward: float
Actions: np.ndarray
StateValue: Dict[State, float]
ActionValue: Dict[State, np.ndarray]
Policy: Callable[[State], Actions]
DeterministicPolicy: Callable[[State], Action]
print('done')
