import numpy as np

from red_blue_world.Patch import Patch
from typing import Tuple, NamedTuple

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3

BEAN = 0
ONION = 1

class PatchConfig(NamedTuple):
    tag: int
    coordinates: Tuple[int,int]

class GridWorld(Patch):
    def __init__(self, size: int):
        self._size = size
        self._state = np.zeros(2)
        
    def _choose_objects(self):
        object_num = min(max(int((self._size ** 2) * 0.2), 1), self._size ** 2)
        choosen_coords = np.random.choice(self._size ** 2, object_num)
        labels = np.random.choice([BEAN, ONION], size=object_num)
        return choosen_coords, labels
    
    def _get_config(self, choosen_coords, labels):
        config = []
        for i, coord in enumerate(choosen_coords):
            x = coord // self._size
            y = coord % self._size
            config.append(PatchConfig(labels[i], (x, y)))
        return config
    
    def generate(self):
        occupancy = self._choose_objects()
        config_list = self._get_config(*occupancy)   
        return config_list
    
    def take_action(self, state, action) -> Tuple[int, int]:
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
    gw = GridWorld(2)
    
    print(gw.generate())
    print(gw.take_action((0,0), RIGHT))