import numpy as np
import enum

AgentState = np.ndarray
Action = int
Reward = float

class Direction(enum.Enum):
    up = 0
    right = 1
    down = 2
    left = 3

    # this allows consistent types
    # even if we are not actually leaving the patch
    none = 4
