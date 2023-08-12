import numpy as np

from Quilt import Quilt

class RedBlueEnv:
    def __init__(self, config):
        self.env_name = config['env_name']
        self.gird_size = config['grid_size']

        self.quilt = self._init_env(config)

    def _init_env(self, config):
        """ Initialize the environment based on the config. """
        env = Quilt(config)
        return env

    def reset(self):
        """ Reset the environment to the start state. """
        return self.quilt.reset()

    def step(self, action):
        """ Take a step in the environment. """
        """ Quilt will control the load and delete of the patches """
        return self.quilt.step(action)

if __name__ == '__main__':
    config = {
        'env_name': 'gw',
        'grid_size': 5,
        'obs_size': 2
    }
    env = RedBlueEnv(config)
    n=0
    env.reset()

    while n < 1000000:
        action = np.random.choice([0, 1, 2, 3])
        output = env.step(action)
        n += 1
        # print(action)
        # print(output[0])
        # print(output[1])
        # input()