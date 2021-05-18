USE_GPU = False
N_EPISODE = 200  # The number of episodes for training
MAX_STEP = 1000  # The number of steps for each episode
BATCH_SIZE = 32  # The number of experiences for each replay
MEMORY_SIZE = 100000  # The size of the batch for storing experiences
# After this number of episodes, the DQN model is saved for testing later.
SAVE_NETWORK = 10
# The number of experiences are stored in the memory batch before starting replaying
INITIAL_REPLAY_SIZE = 100
INPUTNUM = 10000  # The number of input values for the DQN model
ACTIONNUM = 10  # The number of actions output from the DQN model
MAP_MAX_X = 21  # Width of the Map
MAP_MAX_Y = 9  # Height of the Map
