# This shouldn't need to be a class. No need to instantiate
# This should handle _when_ to load and _what_ to load
# This should also handle unloading
# Should be punted to a separate thread
#   - Thread, not process. Let the GIL be our locking mechanism
#   - Should be io bound and not compute bound

from red_blue_world.patches.gw import ContinualGridWorld
from red_blue_world.patches.pickyeater import ContinualCollectRGB, ContinualCollectPartial
from red_blue_world.interfaces import AgentState, Direction

# hard coding the size
# TODO: should change this later
SIZE = 15

def patch_loader(name, agent_loc):
    if name == 'gw':
        size = SIZE #some number
        return ContinualGridWorld(size, agent_loc)
    
def transit_agent(d: Direction, agent_loc: AgentState):
    env_x, env_y = SIZE, SIZE
    x, y = agent_loc
    if d == Direction.up:
        next_loc = (x, 0)
    elif d == Direction.down:
        next_loc = (x, env_y - 1)
    elif d == Direction.left:
        next_loc = (env_x - 1, y)
    else:
        assert d == Direction.right
        next_loc = (0, y)
    return next_loc