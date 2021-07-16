import Parameter as param
import numpy as np


class AgentKnowledge:
    def __init__(self, idx, location_list=None,
                 moving_dir_list=None,
                 max_x=param.max_x, min_x=param.min_x, max_y=param.max_y, min_y=param.min_y,
                 n_size=param.n_size) -> None:
        self.id = idx
        self.location_list = location_list
        self.moving_dir_list = moving_dir_list
        self.max_x = max_x
        self.min_x = min_x
        self.max_y = max_y
        self.min_y = min_y
        self.n_size = n_size

    def update_infos(self, locs, dirs):
        self.location_list = locs
        self.moving_dir_list = dirs

    def build_matrix_state(self, delta_time):
        matrix_state = np.zeros((self.n_size, self.n_size))

        unit_x = (self.max_x - self.min_x) / self.n_size
        unit_y = (self.max_y - self.min_y) / self.n_size

        for idx, (x, y) in enumerate(self.location_list):
            if idx == self.id:
                continue
            x += self.moving_dir_list[idx][0] * delta_time
            x += self.moving_dir_list[idx][1] * delta_time
            idx_x = int(x / unit_x)
            idx_y = int(y / unit_y)

            if (idx_x > 0 and idx_x < self.n_size and idx_y > 0 and idx_y < self.n_size):
                matrix_state[idx_x, idx_y] = 1
                self.location_list[idx] = x, y

        return matrix_state
