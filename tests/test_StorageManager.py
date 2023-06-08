import unittest
import os
import json
import shutil
from red_blue_world.StorageManager import StoreFactory
from typing import NamedTuple
import simplejson

class TestPatchConfig(NamedTuple):
    label: int
    x: int
    y: int

#TODO: switch into pytest, maybe have fixtures
# also tbh its not great that the tests depend on the store function ... might want to adjust that
class TestSqliteBasicStorage(unittest.TestCase):

    def test_load_patch_state(self):

        sqlite_basic = StoreFactory.create_store("sqlite_basic", "load_patch_state_test.db")
        original_patch_state = {"grid_weather" : "hot", "cells" : [TestPatchConfig(label=0, x=6, y=8), TestPatchConfig(label=8, x=9, y=4)]} 
        sqlite_basic.store_patch(patch_id="abc123", patch_state=original_patch_state)
        patch_state_stored = sqlite_basic.load_patch_state(patch_id="abc123")
        self.assertDictEqual(patch_state_stored, simplejson.loads(simplejson.dumps(original_patch_state)))

    def test_load_patch_states(self):

        sqlite_basic = StoreFactory.create_store("sqlite_basic", "load_patch_states_test.db")
        original_patch_state_1 = {"grid_weather" : "hot", "cells" : [TestPatchConfig(label=0, x=6, y=8), TestPatchConfig(label=8, x=9, y=4)]} 
        original_patch_state_2 = {"grid_weather" : "cold", "cells" : [TestPatchConfig(label=0, x=6, y=8), TestPatchConfig(label=8, x=9, y=4)]} 

        sqlite_basic.store_patch(patch_id="abc123", patch_state=original_patch_state_1)
        sqlite_basic.store_patch(patch_id="def456", patch_state=original_patch_state_2)

        patch_states_stored = sqlite_basic.load_patch_states(patch_ids=["abc123", "def456"])
        
        self.assertTupleEqual(("abc123", simplejson.loads(simplejson.dumps(original_patch_state_1))), patch_states_stored[0])
        self.assertTupleEqual(("def456", simplejson.loads(simplejson.dumps(original_patch_state_2))), patch_states_stored[1])

    def test_patch_exists(self):

        sqlite_basic = StoreFactory.create_store("sqlite_basic", "patch_exists_test.db")
        original_patch_state = {"grid_weather" : "hot", "cells" : [TestPatchConfig(label=0, x=6, y=8), TestPatchConfig(label=8, x=9, y=4)]} 
        sqlite_basic.store_patch(patch_id="abc123", patch_state=original_patch_state)
        exists_1 = sqlite_basic.patch_exists(patch_id="abc123")
        exists_2 = sqlite_basic.patch_exists(patch_id="def456")
        self.assertTrue(exists_1)
        self.assertFalse(exists_2)
