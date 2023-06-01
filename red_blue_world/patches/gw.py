import numpy as np

from red_blue_world.Patch import Patch
from typing import Tuple, NamedTuple

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3

FLOWER = 0
WEED = 1

OBJECT_PERCENTAGE = 0.1

class PatchConfig(NamedTuple):
    """
        A named tuple that represents the configuration of a patch coordinates.
    """
    label: int
    x: int
    y: int

class GridWorld(Patch):
    def __init__(self, size: int):
        self._size = size
        self._state = np.zeros(2)
        
        # getting the total number of cells in the grid
        self._cell_num = self._size ** 2
        
    def _choose_objects(self):
        """
            Returns a tuple of coordinates and corresponding labels of both jelly beans and onions.
        """
        object_num = min(max(int(self._cell_num * OBJECT_PERCENTAGE), 1), self._cell_num)
        choosen_coords = np.random.choice(self._cell_num, object_num)
        labels = np.random.choice([FLOWER, WEED], size=object_num)
        return choosen_coords, labels
    
    def _to_coords(self, coord: int) -> Tuple[int, int]:
        """
            Returns the coordinates of the cell.
        """
        print(coord)
        x = coord % self._size
        y = coord // self._size
        return x, y
    
    def _get_config(self, choosen_coords: np.ndarray, labels: np.ndarray):
        """
            Returns a list of PatchConfig objects where each corresponds to 
            the object label and coordinates.
        """
        config = {}
        for coord_idx, label in zip(choosen_coords, labels):
            coords = self._to_coords(coord_idx)
            config[coords] = PatchConfig(label, *coords)
        return config
    
    def generate(self):
        """
            Generates the grid world by adding the objects to the grid and 
            returns a list of PatchConfig objects.
        """
        occupancy = self._choose_objects()
        config_list = self._get_config(*occupancy)   
        return config_list
    
    @staticmethod
    def take_action(state: tuple, action: int) -> Tuple[int, int]:
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
    grid = GridWorld(3)
    config = grid.generate()
    print(config)
    # print(grid.take_action((0, 0), RIGHT))
    # print(grid.take_action((0, 0), DOWN))
    # print(grid.take_action((0, 0), LEFT))
    # print(grid.take_action((0, 0), UP))
    # print(grid.take_action((0, 0), 4))