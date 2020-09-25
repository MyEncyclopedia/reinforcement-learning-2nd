from typing import Tuple, Dict, Callable
import numpy as np


State: Tuple[int, bool, int]
Actions: np.ndarray
ActionValue: Dict[State, np.ndarray]
Policy: Callable[[State], Actions]