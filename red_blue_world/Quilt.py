from itertools import product
from typing import Dict, Tuple
from red_blue_world.Patch import Patch

# this is the coordination of individual patches
# this has a state which is: "which patch is the agent currently in?"
# when a patch says that the agent is leaving, it also says in which direction
#   on the next step, the patchwork informs the next patch that the agent is entering

PatchID = Tuple[int, int]

class Quilt:
    def __init__(self):
        self._patches: Dict[PatchID, Patch] = {}
        self._active_patch_id: PatchID = (0, 0)
        self._active_patch: Patch = ...

    def ensure_load9x9(self):
        for coord in product(range(3), range(3)):
            if coord in self._patches: continue

            # TODO: tell patch loader to grab patch ${coord}

    def build_patch(self) -> Patch:
        ...
