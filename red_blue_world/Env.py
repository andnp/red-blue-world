import numpy as np

from patches.gw import ContinualGridWorld
from patches.pickyeater import ContinualCollectPartial, ContinualCollectRGB, ContinualCollectXY


class RedBlueEnv:
    def __init__(self, config):
        self.env_name = config['env_name']
        self.gird_size = config['grid_size']

        self.env = self._init_env()

    def _init_env(self):
        """ Initialize the environment based on the config. """

        if self.env_name == 'gw':
            env = ContinualGridWorld(self.gird_size)
        elif self.env_name == 'pe_partial':
            env = ContinualCollectPartial(self.gird_size)
        elif self.env_name == 'pe_rgb':
            env = ContinualCollectRGB(self.gird_size)
        elif self.env_name == 'pe_xy':
            env = ContinualCollectXY(self.gird_size)

        return env

    def reset(self):
        """ Reset the environment to the start state. """
        return self.env.reset()

    def step(self, action):
        """ Take a step in the environment. """
        
        # TODO: make new_patch and direction the same type (the last element of output)
        return self.env.step(action)


if __name__ == '__main__':
    config = {
        'env_name': 'pe_partial',
        'grid_size': 5
    }

    np.random.seed(0)

    env = RedBlueEnv(config)
    state, observation = env.reset()
    done = False
    while not done:
        action = int(input('input_action: '))
        state, observation, reward, done, new_patch = env.step(action)
        
        print(f'state: {state}, observation: {observation}, reward: {reward}, done: {done}, new_patch: {new_patch}')

        if new_patch:
            break
