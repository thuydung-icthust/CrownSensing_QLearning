from random import random, randrange
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
import numpy as np
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import Parameter as param
# import keras.backend.tensorflow_backend as K

# Deep Q Network off-policy


class DQN:
    def __init__(
        self,
        num_agent,
        action_space,  # The number of actions for the DQN network
        input_dim=param.input_dim,  # The number of inputs for the DQN network
        gamma=0.95,  # The discount factor
        epsilon=1,  # Epsilon - the exploration factor
        epsilon_min=0.1,  # The minimum epsilon
        epsilon_decay=0.999,  # The decay epislon for each update_epsilon time
        learning_rate=0.00025,  # The learning rate for the DQN network
        tau=0.125,  # The factor for updating the DQN target network from the DQN network
        model=None,  # The DQN model
        target_model=None,  # The DQN target model
        sess=None
    ):
        self.num_agent = num_agent
        self.action_space = action_space
        self.input_dim = input_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau

        # Create networks
        self.models = self.create_models()
        self.target_models = self.create_models()

    def create_models(self):
        models = []
        for i in range(0, self.num_agent):
            model = Sequential()
            model.add(Dense(100, input_dim=self.input_dim))
            model.add(Activation('relu'))
            model.add(Dense(50))
            model.add(Activation('relu'))
            model.add(Dense(self.action_space))
            model.add(Activation('linear'))
            sgd = optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.95)
            model.compile(optimizer=sgd, loss='mse')
            models.append(model)
        return models

    def act(self, state):
        actions = []
        for idx in range(len(self.models)):
            a_max = np.argmax(self.models[idx].predict(state[idx].reshape(1, self.input_dim)))
            a_chosen = 0
            if (random() < self.epsilon):
                a_chosen = randrange(self.action_space)
            else:
                a_chosen = a_max
            actions.append(a_chosen)
        return np.array(actions)

    def replay(self, samples, batch_size):
        # print(samples)
        for idx in range(len(self.models)):
            inputs = np.zeros((batch_size, self.input_dim))
            targets = np.zeros((batch_size, self.action_space))

            for i in range(0, batch_size):
                state = samples[0][i]
                action = samples[1][i, idx]
                reward = samples[2][i]
                new_state = samples[3][i]
                done = samples[4][i]

                inputs[i, :] = state
                # print(str(i) + ' ' + str(idx))
                # print(state)
                # print(action)
                # print(reward)
                # print(new_state)
                # print(done)
                targets[i, :] = self.target_models[idx].predict(
                    state.reshape(1, -1))
                if done:
                    # if terminated, only equals reward
                    targets[i, action] = reward
                else:
                    Q_future = np.max(self.target_models[idx].predict(
                        new_state.reshape(1, -1)))
                    targets[i, action] = reward + Q_future * self.gamma
            # Training
            loss = self.models[idx].train_on_batch(inputs, targets)

    def target_train(self):
        for i in range(len(self.models)):
            weights = self.models[i].get_weights()
            target_weights = self.target_models[i].get_weights()
            for i in range(0, len(target_weights)):
                target_weights[i] = weights[i] * self.tau + \
                    target_weights[i] * (1 - self.tau)

            self.target_models[i].set_weights(target_weights)

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def save_model(self, path, model_name):
        # serialize model to JSON
        for i in range(len(self.models)):
            model_json = self.models[i].to_json()
            with open(path + model_name + '_' + str(i) + ".json", "w") as json_file:
                json_file.write(model_json)
                # serialize weights to HDF5
                self.model.save_weights(path + model_name + '_' + str(i) + ".h5")
                print("Saved model to disk")
