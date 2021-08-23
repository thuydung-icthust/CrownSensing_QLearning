import numpy as np

theta = 0.75
gamma = 0.05
sigma = 0.2

thetab = 0.9
gammab = 0.05
sigmab = 0.05

gamma2 = 0.99  # discount factor for past reward
eps = np.finfo(np.float32).eps.item()  # normalize factor

input_dim = 16
n_size = 100
num_car = 15

cover_radius = 200
cover_time = 60  # To
update_step = 240  # Tmax
step_length = 10

initial_prob = 0.9
record_time_step = 600
min_time_step = 25200
max_time_step = 82800
max_t = max_time_step - min_time_step

mc_approximation = 100000

# grid size
max_x = 1 
max_y = 1 
min_x = -1 
min_y = -1 

random_log_file = "./log/random.txt"
