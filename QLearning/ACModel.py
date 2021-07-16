from AgentKnowledge import AgentKnowledge
import random
import numpy as np
from numpy.matrixlib.defmatrix import matrix
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


class ActorCritic():
    def __init__(self,
                 num_node,
                 input_dim,
                 matrix_input_dim,
                 action_space,
                 gamma=0.99,
                 learning_rate=0.001,
                 hidden_1=128,
                 hidden_2=64):  # The learning rate for the DQN network
        self.num_node = num_node
        self.input_dim = input_dim
        self.matrix_input_dim = matrix_input_dim
        self.output_dim = action_space
        self.gamma = gamma

        self.learning_rate = learning_rate
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2

        self.target_model = self.create_model()
        self.agent_models = self.create_models()
        self.knowledges = [AgentKnowledge(idx) for idx in range(self.num_node)]
        self.action_steps = [30 for i in range(self.num_node)]

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_func = keras.losses.Huber()

    def create_model(self):
        self_input = layers.Input(shape=(self.input_dim,))
        matrix_input = layers.Input(shape=(self.matrix_input_dim, self.matrix_input_dim, 1))
        # self info process
        hid1 = layers.Dense(self.hidden_1, activation="relu")(self_input)
        hid2 = layers.Dense(self.hidden_2, activation="relu")(hid1)

        # matrix process
        conv1 = layers.Conv2D(self.hidden_1, kernel_size=3, activation="relu")(matrix_input)
        conv2 = layers.Conv2D(self.hidden_2, kernel_size=3, activation="relu")(conv1)
        pooling1 = layers.MaxPool2D()(conv2)
        flat1 = layers.Flatten()(pooling1)

        concat1 = layers.concatenate([flat1, hid2], axis=1)
        hid3 = layers.Dense(self.hidden_2, activation='relu')(concat1)

        action = layers.Dense(self.output_dim, activation="softmax")(
            hid3)  # predict the prob distribution for actions given state
        critic = layers.Dense(1)(hid3)  # predict the values for each actor

        model = keras.Model(inputs=[self_input, matrix_input], outputs=[action, critic])

        return model

    def create_models(self):
        models = []
        for i in range(self.num_node):
            model = self.create_model()
            models.append(model)

        return models

    def forward(self, node_state, map_state, idx):
        # actions, value = self.agent_models[idx]([node_state, map_state])
        # print('GO HERE 2')
        actions, value = self.target_model([node_state, map_state])
        return actions, value

    def update_weights(self):
        target_weights = self.target_model.get_weights()

        # copy weight from target weights to agent models
        for i in range(self.num_node):
            self.agent_models[i].set_weights(target_weights)

    def act(self, action_props):
        a_chosen = np.random.choice(self.output_dim, p=np.squeeze(action_props))

        return a_chosen

    def backprop(self, history, tape):
        actor_losses = []
        critic_losses = []

        for log_prob, value, ret in history:
            diff = ret - value
            actor_losses.append(-log_prob * diff)

            critic_losses.append(self.loss_func(tf.expand_dims(ret, 0), tf.expand_dims(value, 0)))

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        print(loss_value)
        grads = tape.gradient(loss_value, self.target_model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.target_model.trainable_variables))

    def save_network(self, ep):
        save_path = 'checkpoint/ac_ep_{}.h5'
        self.target_model.save_weights(save_path.format(ep))

    def load_model(self, path):
        self.target_model.load_weights(path)


if __name__ == '__main__':
    model = ActorCritic(15, 4, 100, 2)

    acts, critic = model.forward(tf.convert_to_tensor(np.random.rand(1, 4)),
                                 tf.convert_to_tensor(np.random.rand(1, 100, 100)), 1)

    print(acts)
    print(critic)

    # acts_2 = model.act(acts)
    # print(acts_2)
    # print(acts[0][0, acts_2[0]])
