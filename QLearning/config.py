N_EPISODE = 1001  # The number of episodes for training
MAX_STEP = 3000  # The number of steps for each episode
BATCH_SIZE = 32  # The number of experiences for each replay
MEMORY_SIZE = 10000  # The size of the batch for storing experiences
# After this number of episodes, the DQN model is saved for testing later.
SAVE_NETWORK = 50
# The number of experiences are stored in the memory batch before starting replaying
INITIAL_REPLAY_SIZE = 50

NUM_NODE = 15
INPUTNUM = 4  # The number of input values for the DQN model
ACTIONNUM = 2  # The number of actions output from the DQN model
MAP_MAX_X = 21  # Width of the Map
MAP_MAX_Y = 9  # Height of the Map
