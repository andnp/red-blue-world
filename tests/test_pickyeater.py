import copy
import unittest

import numpy as np
import sys
sys.path.insert(0, '..')

from red_blue_world.patches.pickyeater import ContinualCollectRGB, ContinualCollectPartial, draw

class TestConfig(unittest.TestCase):
    # def test_step(self, test_steps=10):
    #     # env = ContinualCollectRGB()
    #     env = ContinualCollectPartial()
    #     state = env.reset()
    #     draw(state)
    #     for _ in range(test_steps):
    #         action = int(input('input_action: '))
    #         state, reward, direction = env.step(action)
    #         env.on_enter(state)
    #         draw(state)
    #         print(reward)

    def test_patch(self):
        np.random.seed(0)
        env = ContinualCollectRGB(id='(0,0)')
        state = env.reset()
        # draw(state)
        for _ in range(100):
            action = np.random.randint(env.get_action_dim())
            state, obs, reward, exiting, direction = env.step(action)
            env.on_enter(state)
        pre_save = copy.deepcopy(state)

        obj = env.serialize()
        env.load(obj)

        after_load = env.last_agent_state
        self.assertTrue(np.array_equal(pre_save, after_load), "Inconsistent last_agent_state!")

    def test_fruit_reset(self):
        np.random.seed(0)
        env = ContinualCollectRGB(id='(0,0)')
        state = env.reset()

        """Hack the environment"""
        rb = env.rewarding_blocks
        for b in rb:
            idx = env.object_coords.index(b)
            env.object_status[idx] = 0
        self.assertTrue(env.check_fruit_resetting(), "Should add fruit now")
        self.assertTrue(env.object_status.sum() == len(env.object_status), "All object_status should be 1")

        np.random.seed(0)
        env = ContinualCollectRGB(id='(0,0)')
        state = env.reset()
        pb = env.penalty_blocks
        for b in pb:
            idx = env.object_coords.index(b)
            env.object_status[idx] = 0
        self.assertTrue(not env.check_fruit_resetting(), "Should Not add fruit now")
        self.assertTrue(env.object_status.sum() == len(env.object_status) // 2, "Only rewarding object_status should be 1")

        np.random.seed(0)
        env = ContinualCollectRGB(id='(0,0)')
        state = env.reset()
        rb = env.rewarding_blocks
        pb = env.penalty_blocks
        for b in pb[:len(pb)//2]:
            idx = env.object_coords.index(b)
            env.object_status[idx] = 0
        for b in rb:
            idx = env.object_coords.index(b)
            env.object_status[idx] = 0
        self.assertTrue(env.check_fruit_resetting(), "Should add fruit now")
        self.assertTrue(env.object_status.sum() == len(env.object_status), "All object_status should be 1")
