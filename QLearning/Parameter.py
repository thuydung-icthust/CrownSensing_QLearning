import numpy as np

theta = 0.5
gamma = 0.2
sigma = 0.3

thetab = 0.9
gammab = 0.05
sigmab = 0.05

gamma2 = 0.99  # discount factor for past reward
eps = np.finfo(np.float32).eps.item()  # normalize factor

input_dim = 16
n_size = 100
num_car = 15

cover_radius = 0.6
cover_time = 60  # To
update_step = 240  # Tmax
step_length = 10

initial_prob = 0.9
record_time_step = 600
min_time_step = 25200
max_time_step = 82800
max_t = max_time_step - min_time_step

mc_approximation = 50000

# grid size
max_x = 1 + cover_radius
max_y = 1 + cover_radius
min_x = -1 - cover_radius
min_y = -1 - cover_radius

random_log_file = "./log/random.txt"
