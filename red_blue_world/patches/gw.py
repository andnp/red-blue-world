import numpy as np

from red_blue_world.Patch import Patch
from typing import Tuple, NamedTuple

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3

BEAN = 0
ONION = 1

OBJECT_PERCENTAGE = 0.1

class PatchConfig(NamedTuple):
    label: int
    x: int
    y: int

class GridWorld(Patch):
    def __init__(self, size: int):
        self._size = size
        self._state = np.zeros(2)
        
        self._cell_num = self._size ** 2
        
    def _choose_objects(self):
        """
            Returns a tuple of coordinates and corresponding labels of both jelly beans and onions.
        """
        object_num = min(max(int(self._cell_num * OBJECT_PERCENTAGE), 1), self._cell_num)
        choosen_coords = np.random.choice(self._cell_num, object_num)
        labels = np.random.choice([BEAN, ONION], size=object_num)
        return choosen_coords, labels
    
    def _get_config(self, choosen_coords: np.ndarray, labels: np.ndarray):
        """
            Returns a list of PatchConfig objects where each corresponds to 
            the object label and coordinates.
        """
        config = np.zeros(len(choosen_coords), dtype=PatchConfig)
        for i, coord in enumerate(choosen_coords):
            x = coord // self._size
            y = coord % self._size
            config[i] = PatchConfig(labels[i], x, y)
        return config
    
    def generate(self):
        """
            Generates the grid world by adding the objects to the grid and 
            returns a list of PatchConfig objects.
        """
        occupancy = self._choose_objects()
        config_list = self._get_config(*occupancy)   
        return config_list
    
    def take_action(self, state, action) -> Tuple[int, int]:
        """
            Takes an action and returns the new state.
        """
        x, y = state
        if action == RIGHT:
            x += 1
        elif action == DOWN:
            y -= 1
        elif action == LEFT:
            x -= 1
        elif action == UP:
            y += 1
        else:
            raise Exception(f'Unknown action: {action}')

        return x, y
        
    
if __name__ == '__main__':
    gw = GridWorld(20)
    
    print(gw.generate())
    print(gw.take_action((0,0), RIGHT))