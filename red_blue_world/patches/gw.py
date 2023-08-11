import numpy as np
import enum
import pygame
from typing import Tuple, NamedTuple

from red_blue_world.Patch import Patch
from red_blue_world.interfaces import Direction


OBJECT_PERCENTAGE = 0.1
FLOWER = 1
WEED = 2
AGENT = 3


class Action(enum.Enum):
    up = 0
    right = 1
    down = 2
    left = 3
    stay = 4


class PatchConfig(NamedTuple):
    """
        A named tuple that represents the configuration of a patch coordinates.
    """
    label: int
    x: int
    y: int


class ContinualGridWorld(Patch):
    def __init__(self, size: int, agent_loc: Tuple[int, int] = None):
        self._size = size
        self._state = np.zeros(2)
        self.agent_loc = agent_loc

        # getting the total number of cells in the grid
        self._cell_num = self._size ** 2

        self._action_dim = 5
        self._actions = {
            Action.up.value: (0, 1),
            Action.down.value: (0, -1),
            Action.left.value: (-1, 0),
            Action.right.value: (1, 0),
            Action.stay.value: (0, 0)
        }

        self._dir_mapping = {
            Action.right.value: Direction.right,
            Action.down.value: Direction.down,
            Action.left.value: Direction.left,
            Action.up.value: Direction.up
        }

        self._rewards = {
            FLOWER: 1,
            WEED: -1
        }

        self.config = self._get_config()

    def _choose_objects(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Returns a tuple of coordinates and corresponding labels of both jelly beans and onions. """
        object_num = min(
            max(int(self._cell_num * OBJECT_PERCENTAGE), 1), self._cell_num)
        choosen_coords = np.random.choice(self._cell_num, object_num)
        labels = np.random.choice(
            [FLOWER, WEED], size=object_num)
        return choosen_coords, labels

    def _to_coords(self, coord: int) -> Tuple[int, int]:
        """ Returns the coordinates of the cell. """
        x = coord % self._size
        y = coord // self._size
        return x, y

    def _to_idx(self, x: int, y: int) -> int:
        return x + y * self._size

    def _get_random_coordinate(self) -> Tuple[int, int]:
        """ Randomly chooses a coordinate that is not occupied. """
        rand_state = np.random.randint(low=0, high=self._cell_num)
        while rand_state in self.config.keys():
            rand_state = np.random.randint(low=0, high=self._cell_num)

        return self._to_coords(rand_state)

    def _get_config(self) -> dict:
        """ Returns a dict of PatchConfig objects where each corresponds to 
        the object label and coordinates and the keys are unnormalized indices. """
        choosen_coords, labels = self._choose_objects()

        config = {}
        for coord_idx, label in zip(choosen_coords, labels):
            coords = self._to_coords(coord_idx)
            config[coord_idx] = PatchConfig(label, *coords)
        return config

    def reset(self) -> None:
        """ Should only call this function once, at the very beginning of each run
        to give the strt position of the agent. """
        rand_state = self._get_random_coordinate()
        self.agent_loc = rand_state
        return self.generate_state(), self.generate_observation()

    def generate_state(self) -> tuple:
        """ Getting the state of the grid world. """
        return np.array(self.agent_loc)

    def get_reward(self) -> int:
        """ Getting the reward of the grid world. """
        current_idx = self._to_idx(*self.agent_loc)
        current_state = self.config.get(current_idx)

        return self._rewards[current_state.label] if current_state else 0

    def get_action_dim(self) -> int:
        """ Getting the action dimension of the grid world."""
        return self._action_dim

    def generate_observation(self) -> np.ndarray:
        """ Getting the observation of the grid world. """
        grid = np.zeros((self._size, self._size))
        for value in self.config.values():
            grid[value.x, value.y] = value.label

        grid[self.agent_loc[0], self.agent_loc[1]] = AGENT
        return grid

    def _check_bounds(self, x, y) -> Direction:
        return 0 <= x < self._size and 0 <= y < self._size

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Direction]:
        """ Taking a step in the grid world environment according to given action. """
        direction = self.take_action(action)

        # Ensuring the next position is within bounds
        reward = self.get_reward()
        state = self.generate_state()
        observation = self.generate_observation()

        return state, observation, np.asarray(reward), direction

    def take_action(self, action: int):
        """ Takes an action and returns the new state. """
        x, y = self.agent_loc
        if action in self._actions.keys():
            x += self._actions[action][0]
            y += self._actions[action][1]
        else:
            raise Exception(f'Unknown action: {action}')

        if self._check_bounds(x, y):
            self.agent_loc = x, y
            return Direction.none

        return self._dir_mapping[action]


def visualize(observation: np.ndarray):
    white = (255, 255, 255)
    black = (0, 0, 0)

    grid_size = observation.shape[0]

    pygame.init()
    pygame.font.init()
    pygame.display.set_caption('Gridworld')

    block_size = 100
    window = pygame.display.set_mode(
        (block_size * grid_size, block_size * grid_size))
    font = pygame.freetype.SysFont("seguisym.ttf", size=int(block_size * 0.7))

    done = False
    while done:
        window.fill(white)

        # getting the black lines
        for y in range(grid_size):
            for x in range(grid_size):
                rect = pygame.Rect(x*block_size, y*block_size,
                                   block_size, block_size)
                pygame.draw.rect(window, black, rect, 1)

                if observation[y][x] == FLOWER:
                    emoji = 'F'
                    color = (255, 0, 0)
                elif observation[y][x] == WEED:
                    emoji = 'W'
                    color = (0, 255, 0)
                elif observation[y][x] == AGENT:
                    emoji = 'A'
                    color = (0, 0, 255)
                else:
                    continue

                font.render_to(window, ((y * block_size) + 20,
                                        (x * block_size) + 20), emoji, color)

        # Draws the surface object to the screen.
        pygame.display.update()

        for event in pygame.event.get():
            pygame.time.wait(500000)

            if event.type == pygame.QUIT:
                done = True
                break
