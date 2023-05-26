from typing import Any, Dict, NamedTuple

'''
Each Patch element has 
1. a tag that can be 0 for onions or  1 for Jelly
2. a tuple that contains the corrdinates of that point. 
'''
class PatchConfig(NamedTuple):
    tag: int 
    coordinates: tuple
    
