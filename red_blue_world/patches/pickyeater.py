import numpy as np
import matplotlib.pyplot as plt
from red_blue_world.Patch import *

class CCPatch(Patch):
    def __init__(self):
        super(CCPatch, self).__init__()
        self.last_agent_state = None

    def load(self, patch_state: PatchState) -> None:
        self.rewarding_color = patch_state['rewarding_color']
        self.rewarding_blocks = patch_state['rewarding_blocks']
        self.penalty_color = patch_state['penalty_color']
        self.penalty_blocks = patch_state['penalty_blocks']
        self.reds = patch_state['reds']
        self.blues = patch_state['blues']
        self.object_status = patch_state['object_status']
        self.agent_loc = patch_state['agent_loc']
        self.last_agent_state = patch_state['last_agent_state']
        return
    
    def on_enter(self, last_agent_state: np.ndarray) -> None:
        self.last_agent_state = last_agent_state
        return
    
    def serialize(self) -> PatchState:
        object = {
            "rewarding_color": self.rewarding_color,
            "rewarding_blocks": self.rewarding_blocks,
            "penalty_color": self.penalty_color,
            "penalty_blocks": self.penalty_blocks,
            "reds": self.reds,
            "blues": self.blues,
            "object_status": self.object_status,
            "agent_loc": self.agent_loc,
            "last_agent_state": self.last_agent_state
        }
        return object


class ContinualCollectXY(CCPatch):
    """
    Remove terminal state
    Remove green fruit, use 2 colors
    Remove penalty each timestep
    Reset fruit when there is no more fruit to pick
    """
    def __init__(self, seed=np.random.randint(int(1e5))):
        super(ContinualCollectXY, self).__init__()
        self.object_coords = [(7, 2), (2, 7), (8, 6), (6, 8),
                              (8, 0), (0, 8), (14, 0), (0, 14),
                              (6, 14), (14, 6), (7, 11), (11, 7)]
        np.random.seed(seed)

        # one indiciate the object is available to be picked up
        self.object_status = np.ones(len(self.object_coords))
        self.action_dim = 4

        self.obstacles_map = self.get_obstacles_map()
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)] # right, down, left, up, stay

        self.agent_loc = (0, 0)
        self.object_status = np.ones(len(self.object_coords))

        self.min_x, self.max_x, self.min_y, self.max_y = 0, 14, 0, 14

        self.blues, self.reds = None, None
        self.rewarding_color = 'red'
        self.rewarding_blocks = None
        self.penalty_color = 'blue'
        self.penalty_blocks = None
        
    def get_action_dim(self):
        return len(self.actions)
    
    def info(self, key):
        return

    def generate_state(self, agent_loc, object_status, reds, blues):
        reds = np.array(reds).flatten()
        blues = np.array(blues).flatten()
        return np.concatenate([np.array(agent_loc), object_status, reds, blues])

    def generate_observation(self, agent_loc, object_status, reds, blues):
        reds = np.array(reds).flatten()
        blues = np.array(blues).flatten()
        return np.concatenate([np.array(agent_loc), object_status, reds, blues])

    def reset(self):
        """
        Should only call this function once, at the very beginning of each run
        """
        self.reset_fruit()
        while True:
            rand_state = np.random.randint(low=0, high=15, size=2)
            rx, ry = rand_state
            if not int(self.obstacles_map[rx][ry]) and \
                    not (rx, ry) in self.object_coords:
                self.agent_loc = rx, ry
                return self.generate_state(self.agent_loc, self.object_status, self.reds, self.blues), \
                       self.generate_observation(self.agent_loc, self.object_status, self.reds, self.blues)
    
    def reset_fruit(self):
        obj_ids = np.arange(len(self.object_coords))
        obj_ids = np.random.permutation(obj_ids)

        red_ids, blue_ids = obj_ids[:len(obj_ids)//2], obj_ids[len(obj_ids)//2:]

        self.reds = [self.object_coords[k] for k in red_ids]
        self.blues = [self.object_coords[k] for k in blue_ids]
        self.rewarding_color = np.random.choice(['red', 'blue'])
        if self.rewarding_color == 'red':
            self.rewarding_blocks = self.reds
            self.penalty_blocks = self.blues
        elif self.rewarding_color == 'blue':
            self.rewarding_blocks = self.blues
            self.penalty_blocks = self.reds
        else:
            raise NotImplementedError
        self.object_status = np.ones(len(self.object_coords))
        return

    def check_fruit_resetting(self):
        non_rewarding = True
        for [x, y] in self.rewarding_blocks:
            object_idx = self.object_coords.index((x, y))
            if self.object_status[object_idx]:
                non_rewarding = False
        if non_rewarding:
            self.reset_fruit()
        return non_rewarding

    def step(self, a):
        dx, dy = self.actions[a]
        x, y = self.agent_loc

        nx = x + dx
        ny = y + dy

        # Ensuring the next position is within bounds
        nx, ny = min(max(nx, self.min_x), self.max_x), min(max(ny, self.min_y), self.max_y)
        if not self.obstacles_map[nx][ny]:
            x, y = nx, ny
            
        if x == self.agent_loc[0] and y == self.agent_loc[1]:
            direction = 4 # stay
        else:
            direction = a

        reward = 0.0
        if (x, y) in self.object_coords:
            object_idx = self.object_coords.index((x, y))
            if self.object_status[object_idx]:
                # the object is available for picking
                self.object_status[object_idx] = 0.0
                if (x, y) in self.rewarding_blocks:
                    reward += 1.0
                elif (x, y) in self.penalty_blocks:
                    reward += -1.0
        
        self.agent_loc = x, y
        self.check_fruit_resetting()

        state = self.generate_state(self.agent_loc, self.object_status, self.reds, self.blues)
        observation = self.generate_observation(self.agent_loc, self.object_status, self.reds, self.blues)
        return state, observation, np.asarray(reward), np.asarray(False), direction

    def get_visualization_segment(self):
        raise NotImplementedError

    def get_obstacles_map(self):
        _map = np.zeros([15, 15])
        _map[7, 0:2] = 1.0
        _map[7, 4:11] = 1.0
        _map[7, 13:] = 1.0

        _map[0:2, 7] = 1.0
        _map[4:11, 7] = 1.0
        _map[13:, 7] = 1.0

        return _map


class ContinualCollectRGB(ContinualCollectXY):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)
        np.random.seed(seed)
        d = len(self.obstacles_map)
        self.state_dim = (d, d, 3)
        self.main_template = np.zeros(self.state_dim)
        for x in range(d):
            for y in range(d):
                if self.obstacles_map[x][y]:
                    self.main_template[x][y] = np.array([0, 0, 0])
                else:
                    self.main_template[x][y] = np.array([128, 128, 128])

        self.episode_template = None

    def get_episode_template(self, reds, blues):
        episode_template = np.copy(self.main_template)
        for rx, ry in reds:
            episode_template[rx][ry] = np.array([255, 0, 0])
        for bx, by in blues:
            episode_template[bx][by] = np.array([0, 0, 255])
        return episode_template

    def reset_fruit(self):
        super(ContinualCollectRGB, self).reset_fruit()
        self.episode_template = self.get_episode_template(self.reds, self.blues)

    def generate_observation(self, agent_loc, object_status, reds, blues):
        state = np.copy(self.episode_template)
        x, y = agent_loc

        for object_idx, coord in enumerate(self.object_coords):
            if not self.object_status[object_idx]:
                ox, oy = coord
                state[ox][oy] = np.array([128, 128, 128])
        state[x][y] = np.array([255, 255, 0])
        return state

    def get_useful(self, img=None):
        raise NotImplementedError

    def get_visualization_segment(self):
        if self.episode_template is not None:
            obj_ids = np.arange(len(self.object_coords))
            obj_ids = np.random.permutation(obj_ids)
            red_ids, blue_ids = obj_ids[:4], obj_ids[4:]
            self.reds = [self.object_coords[k] for k in red_ids]
            self.blues = [self.object_coords[k] for k in blue_ids]

            state_coords = [[x, y] for x in range(15)
                           for y in range(15) if not int(self.obstacles_map[x][y])]
            states = [self.generate_observation(coord, self.object_status, self.reds) for coord in state_coords]
            return np.array(states), np.array(state_coords)
        else:
            raise NotImplementedError

class ContinualCollectPartial(ContinualCollectRGB):
    def __init__(self, seed=np.random.randint(int(1e5))):
        super().__init__(seed)

    def generate_observation(self, agent_loc, object_status, reds, blues):
        state = super().generate_observation(agent_loc, object_status, reds, blues)
        x, y = agent_loc
        if (x < 7 and y < 7):
            visual = [[i,j] for i in range(0, 8) for j in range(0, 8)]
        elif (x > 7 and y < 7):
            visual = [[i,j] for i in range(7, 15) for j in range(0, 8)]
        elif (x < 7 and y > 7):
            visual = [[i,j] for i in range(0, 8) for j in range(7, 15)]
        elif (x > 7 and y > 7):
            visual = [[i,j] for i in range(7, 15) for j in range(7, 15)]
        elif (x == 7 and y < 7):
            visual = [[i,j] for i in range(0, 15) for j in range(0, 8)]
        elif (x == 7 and y > 7):
            visual = [[i,j] for i in range(0, 15) for j in range(7, 15)]
        elif (x < 7 and y == 7):
            visual = [[i,j] for i in range(0, 8) for j in range(0, 15)]
        elif (x > 7 and y == 7):
            visual = [[i,j] for i in range(7, 15) for j in range(0, 15)]
        else:
            raise NotImplementedError
        for coord in [[i,j] for i in range(0, 15) for j in range(0, 15)]:
            if coord not in visual:
                state[coord[0], coord[1]] = np.array([128., 128., 128.])
            # else:
            #     print(coord, state[coord])
        return state

        
def draw(state):
    frame = state.astype(np.uint8)
    figure, ax = plt.subplots()
    ax.imshow(frame)
    plt.axis('off')
    plt.show()
    # plt.savefig("../../plot/img/picky_eater.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
def draw_pretty(state):
    frame = state.astype(np.uint8)

    red, green, blue, gray, black = np.array([255., 0., 0.]), np.array([0., 255., 0.]), \
                             np.array([0., 0., 255.]), np.array([128., 128., 128.]), np.array([0., 0., 0.])
    reddish, bluish = np.array([204., 121., 167.]), np.array([0., 158., 115.])
    light_g, light_b = np.array([236., 236., 236.]), np.array([165., 165., 165.])

    reds = np.all(frame == red, axis=2)
    greens = np.all(frame == green, axis=2)
    blues = np.all(frame == blue, axis=2)
    grays = np.all(frame == gray, axis=2)
    blacks = np.all(frame == black, axis=2)

    frame[reds] = reddish
    frame[greens] = bluish
    frame[blues] = light_g
    frame[grays] = light_g
    frame[blacks] = light_b

    figure, ax = plt.subplots()
    ax.imshow(frame, interpolation='nearest')
    plt.axis('off')
    # plt.show()
    plt.savefig("../../plot/img/picky_eater.pdf", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    env = ContinualCollectPartial()
    state, observation = env.reset()
    done = False
    while not done:
        action = int(input('input_action: '))
        state, observation, reward, done, direction = env.step(action)
        draw(observation)
        print(reward, done, direction)