import unittest
import os
import json
import shutil
from red_blue_world.Config import Config

class TestConfig(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree('.tmp')
        except:
            pass

    def test_get(self):
        test_config = {
            'hi': 1,
            'there': 'b',
            'a': {
                'b': {
                    'c': [1, 2, 3],
                }
            }
        }

        os.makedirs('.tmp/', exist_ok=True)
        with open('.tmp/config.json', 'w') as f:
            json.dump(test_config, f)

        config = Config('.tmp/config.json')

        # can grab root items from config
        self.assertTrue(config.get('hi'), 1)
        self.assertTrue(config.get('there'), 'b')
        self.assertRaises(KeyError, lambda: config.get('test'))

        # can grab nested items
        self.assertTrue(config.get('a.b'), { 'c': [1, 2, 3] })
        self.assertTrue(config.get('a.b.c'), [1, 2, 3])
