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
