import numpy as np


alpha = 36.0
beta = 30.0
base = (500.0, 500.0)

theta = 0.6
gamma = 0.1
sigma = 0.3

thetab = 0.9
gammab = 0.1
sigmab = 0.1

gamma2 = 0.99  # discount factor for past reward
eps = np.finfo(np.float32).eps.item()  # normalize factor

depot = (0.0, 0.0)
b = 400.0
b_energy = 0.0
ER = 0.0000001
ET = 0.00000005
EFS = 0.00000000001
EMP = 0.0000000000000013
input_dim = 16
prob = 0.5
n_size = 100
x_bound = [0, 1000]
y_bound = [0, 1000]
num_car = 15
cover_radius = 0.3
cover_time = 30  # second
eng_per_package = 0.02
record_time_step = 600
min_time_step = 25200
max_time_step = 82800
update_step = 30
max_t = 36000
random_log_file = "./log/random.txt"
