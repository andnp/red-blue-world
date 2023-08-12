# This shouldn't need to be a class. No need to instantiate
# This should handle _when_ to load and _what_ to load
# This should also handle unloading
# Should be punted to a separate thread
#   - Thread, not process. Let the GIL be our locking mechanism
#   - Should be io bound and not compute bound

from red_blue_world.patches.gw import ContinualGridWorld
from red_blue_world.patches.pickyeater import ContinualCollectRGB, ContinualCollectPartial
from red_blue_world.interfaces import AgentState, Direction

def patch_loader(env_name, env_size, agent_loc):
    if env_name == 'gw':
        return ContinualGridWorld(env_size, agent_loc)
    
def transit_agent(d: Direction, agent_loc: AgentState, env_size):
    env_x, env_y = env_size, env_size
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