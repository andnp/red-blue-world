from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Dict, Tuple

from red_blue_world.interfaces import Action, AgentState, Direction, Reward
from red_blue_world.Patch import Patch

# this is the coordination of individual patches
# this has a state which is: "which patch is the agent currently in?"
# when a patch says that the agent is leaving, it also says in which direction
#   on the next step, the patchwork informs the next patch that the agent is entering

PatchID = Tuple[int, int]

class Quilt:
    def __init__(self) -> None:
        self._patches: Dict[PatchID, Patch] = {}
        self._active_patch_id: PatchID = (0, 0)
        self._active_patch: Patch = self.build_patch()

        self._t = 0
        self._unload_clocks: Dict[PatchID, int] = {}

        self._back_thread = ThreadPoolExecutor(max_workers=1)

        # ensure the initial patch is cached
        self._patches[self._active_patch_id] = self._active_patch

    def step(self, a: Action) -> Tuple[AgentState, Reward]:
        s, r, d = self._active_patch.step(a)

        # TODO: instead of relying on direction to signal unloading and transitioning
        # use a new signal coming from Patch.step to signal that it is time to transition
        if d != Direction.none:
            self._maybe_unload(self._active_patch_id)
            next_id = self._handle_patch_transition(self._active_patch_id, d)

            self._active_patch_id = next_id
            self._active_patch = self._patches[next_id]

        return (s, r)

    def _handle_patch_transition(self, patch_id: PatchID, d: Direction) -> PatchID:
        if d == Direction.up: next_id = _up(patch_id)
        elif d == Direction.down: next_id = _down(patch_id)
        elif d == Direction.left: next_id = _left(patch_id)
        else:
            assert d == Direction.right
            next_id = _right(patch_id)

        # *synchronously* ensure the next patch is loaded
        # if this is anything more than a no-op, we screwed up somewhere
        self._ensure_load(next_id)

        # asynchronously load the 9x9 square around the next patch
        self._back_thread.submit(self._ensure_load9x9, patch_id)

        return next_id

    def _ensure_load(self, patch_id: PatchID) -> None:
        # shortcut if there is no work to be done
        if patch_id in self._patches: return

        if self.patch_exists(patch_id):
            patch = self.load_patch(patch_id)
        else:
            patch = self.build_patch()

        self._patches[patch_id] = patch

    def _ensure_load9x9(self, patch_id: PatchID) -> None:
        x, y = patch_id

        for dx, dy in product(range(-3, 4), range(-3, 4)):
            coord = (x + dx, y + dy)
            self._ensure_load(coord)

    def _maybe_unload(self, patch_id: PatchID) -> None:
        # always reset the clock for the current patch
        self._unload_clocks[patch_id] = 50

        # assumes that `self._unload_clocks` is usually empty or very small
        # otherwise, this would cause a performance degradation on every tick
        for key in self._unload_clocks:
            self._unload_clocks[key] -= 1

            if self._unload_clocks[key] == 0:
                del self._unload_clocks[key]
                self._back_thread.submit(self.unload_patch, key)

    # -------------------
    # TEMP: type stubs --
    # -------------------

    def load_patch(self, patch_id: PatchID) -> Patch:
        # TODO: this should call out to the patch_loader
        # TODO: probably this can be inlined. Keeping as a type-stub for now
        raise NotImplementedError()

    def unload_patch(self, patch_id: PatchID) -> None:
        # TODO: this should call out to the patch_loader
        # TODO: probably this can be inlined. Keeping as a type-stub for now

        del self._patches[patch_id]
        raise NotImplementedError()

    def patch_exists(self, patch_id: PatchID) -> bool:
        # TODO: this should call out to the patch_loader
        # TODO: probably this can be inlined. Keeping as a type-stub for now
        raise NotImplementedError()

    def build_patch(self) -> Patch:
        raise NotImplementedError()

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
