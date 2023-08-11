from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Dict, Tuple

from red_blue_world.interfaces import Action, EnvState, AgentState, Direction, Reward
from red_blue_world.Patch import Patch
from red_blue_world import patch_loader

# this is the coordination of individual patches
# this has a state which is: "which patch is the agent currently in?"
# when a patch says that the agent is leaving, it also says in which direction
#   on the next step, the patchwork informs the next patch that the agent is entering

PatchID = Tuple[int, int]

class Quilt:
    def __init__(self, config) -> None:
        self.env_name = config['env_name']
        self.env_size = config['grid_size']
        
    def reset(self):
        self._patches: Dict[PatchID, Patch] = {}
        self._active_patch_id: PatchID = (0, 0)
        self._active_patch: Patch = self.build_patch(self.env_name, self.env_size, None)

        self._t = 0
        self.agent_loc = None
        self._unload_clocks: Dict[PatchID, int] = {}

        self._back_thread = ThreadPoolExecutor(max_workers=1)

        # ensure the initial patch is cached
        self._patches[self._active_patch_id] = self._active_patch
        
        return self._active_patch.reset()

    def step(self, a: Action) -> Tuple[EnvState, AgentState, Reward]:
        s, o, r, d = self._active_patch.step(a)
        
        # use direction signal coming from Patch.step to signal that it is time to transition
        if d != Direction.none:
            self._maybe_unload(self._active_patch_id)
            next_id = self._handle_patch_transition(self._active_patch_id, d, s)

            self._active_patch_id = next_id
            self._active_patch = self._patches[next_id]
            self._active_patch.on_enter(s)
            
        return (s, o, r)

    def _handle_patch_transition(self, patch_id: PatchID, d: Direction, agent_loc: AgentState) -> PatchID:
        if d == Direction.up: next_id = _up(patch_id)
        elif d == Direction.down: next_id = _down(patch_id)
        elif d == Direction.left: next_id = _left(patch_id)
        else:
            assert d == Direction.right
            next_id = _right(patch_id)
        next_loc = patch_loader.transit_agent(d, agent_loc, self.env_size)
        print(next_loc)

        # *synchronously* ensure the next patch is loaded
        # if this is anything more than a no-op, we screwed up somewhere
        self._ensure_load(next_id, next_loc)

        # asynchronously load the 3x3 square around the next patch
        self._back_thread.submit(self._ensure_load3x3, (patch_id, next_loc))

        return next_id

    def _ensure_load(self, patch_id: PatchID, agent_loc: AgentState) -> None:
        # shortcut if there is no work to be done
        if patch_id in self._patches: return

        if self.patch_exists(patch_id):
            patch = self.load_patch(patch_id, agent_loc)
        else:
            patch = self.build_patch(self.env_name, self.env_size, agent_loc)

        self._patches[patch_id] = patch

    def _ensure_load3x3(self, patch_id: PatchID, agent_loc: AgentState) -> None:
        x, y = patch_id

        for dx, dy in product(range(-1, 2), range(-1, 2)):
            coord = (x + dx, y + dy)
            self._ensure_load(coord, agent_loc)

    def _ensure_load9x9(self, patch_id: PatchID, agent_loc: AgentState) -> None:
        x, y = patch_id

        for dx, dy in product(range(-3, 4), range(-3, 4)):
            coord = (x + dx, y + dy)
            self._ensure_load(coord, agent_loc)

    def _maybe_unload(self, patch_id: PatchID) -> None:
        # always reset the clock for the current patch
        self._unload_clocks[patch_id] = 50

        # assumes that `self._unload_clocks` is usually empty or very small
        # otherwise, this would cause a performance degradation on every tick
        for key in self._unload_clocks:
            self._unload_clocks[key] -= 1

            if self._unload_clocks[key] == 0:
                self._back_thread.submit(self.unload_patch, key)

    # -------------------
    # TEMP: type stubs --
    # -------------------

    def load_patch(self, patch_id: PatchID, agent_loc) -> Patch:
        # this should call out to the patch_loader
        # probably this can be inlined. Keeping as a type-stub for now
        patch = self._patches[patch_id]
        patch.agent_loc = agent_loc
        return patch

    def unload_patch(self, patch_id: PatchID) -> None:
        # this should call out to the patch_loader
        # probably this can be inlined. Keeping as a type-stub for now
        del self._patches[patch_id]
        del self._unload_clocks[patch_id]

    def patch_exists(self, patch_id: PatchID) -> bool:
        # this should call out to the patch_loader
        # probably this can be inlined. Keeping as a type-stub for now
        return self._patches.get(patch_id, None) is not None

    def build_patch(self, env_name, env_size, agent_loc) -> Patch:
        # Calling patch_loader to initialize a new patch
        return patch_loader.patch_loader(env_name, env_size, agent_loc)

# ------------------------
# -- Internal utilities --
# ------------------------

def _up(coords: PatchID) -> PatchID:
    x, y = coords
    return (x, y + 1)

def _down(coords: PatchID) -> PatchID:
    x, y = coords
    return (x, y - 1)

def _right(coords: PatchID) -> PatchID:
    x, y = coords
    return (x + 1, y)

def _left(coords: PatchID) -> PatchID:
    x, y = coords
    return (x - 1, y)
