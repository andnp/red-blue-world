import numpy as np

from abc import abstractmethod
from typing import Any, Tuple
from red_blue_world.interfaces import AgentState, Action, Reward, Direction

# TODO: make this a nominal type once API has been decided
PatchState = Any

# TODO: this should be a contract/interface for a patch
#  - A patch should be loadable, but is not responsible for deciding when to load
#     - When a patch loads, it is also responsible for setup work at this time
#  - A patch has a state, including "the agent is in this patch"
#  - A patch takes an action, updates state, returns state/reward
#  - A patch also yields "the agent has left in this direction"
#     - Lets assume a 2d patchwork. So agent can leave up,down,left,right
#  - A patch has an "on_enter" method that informs the patch that the agent is entering this patch
#


class Patch:
    @abstractmethod
    def load(self, patch_state: PatchState) -> None:
        ...

    @abstractmethod
    def on_enter(self, last_agent_state: np.ndarray) -> None:
        ...

    @abstractmethod
    def step(self, action: Action) -> Tuple[AgentState, Reward, Direction]:
        ...

    @abstractmethod
    def serialize(self) -> PatchState:
        ...
