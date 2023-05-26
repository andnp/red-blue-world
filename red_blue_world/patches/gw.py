import numpy as np

from red_blue_world.Patch import Patch
from typing import Tuple

class GridWorld(Patch):
    def __init__(self, size: int):
        self._size = size
        self._state = np.zeros(2)
