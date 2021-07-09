import random
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.python.keras.losses import huber


class ActorCritic():
    def __init__(self,
                 num_node,
                 input_dim,
                 action_space,
                 gamma=0.99,
                 epsilon=1,  # Epsilon - the exploration factor
                 epsilon_min=0.1,  # The minimum epsilon
                 epsilon_decay=0.9,  # The decay epislon for each update_epsilon time
                 learning_rate=0.001,
                 hidden_1=128,
                 hidden_2=64):  # The learning rate for the DQN network
        self.num_node = num_node
        self.input_dim = input_dim
        self.output_dim = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2

        self.model = self.create_model()
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_func = keras.losses.Huber()

    def create_model(self):
        inputs = layers.Input(shape=(self.input_dim,))
        hid1 = layers.Dense(self.hidden_1, activation="relu")(inputs)
        hid2 = layers.Dense(self.hidden_2, activation="relu")(hid1)

        outputs = []

        for i in range(self.num_node):
            action = layers.Dense(self.output_dim, activation="softmax")(
                hid2)  # predict the prob distribution for actions given state
            outputs.append(action)

        critic = layers.Dense(self.num_node)(hid2)  # predict the values for each actor
        outputs.append(critic)

        model = keras.Model(inputs=inputs, outputs=outputs)

        return model

    def forward(self, state):
        output = self.model(state)
        return output[:(self.num_node)], output[-1]

    def act(self, action_props):
        actions = []
        for i in range(self.num_node):
            act_probs = action_props[i]
            a_chosen = 0
            if (np.random.ranf() < self.epsilon):
                a_chosen = np.random.randint(0, self.output_dim)
            else:
                a_chosen = np.random.choice(self.output_dim, p=np.squeeze(act_probs))
            actions.append(a_chosen)
        return actions

    def backprop(self, history, tape):
        actor_losses = []
        critic_losses = []

        for log_prop, value, ret in history:
            diff = [ret[i] - value[i] for i in range(self.num_node)]

            actor_losses.append([-log_prop[i] * diff[i] for i in range(self.num_node)])

            critic_losses.append(
                huber(tf.convert_to_tensor(value), tf.convert_to_tensor(ret))
            )

        # Backpropagation
        loss_value = sum([sum(actor_losses[i]) for i in range(len(actor_losses))]) + sum(critic_losses)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def update_epsilon(self):
        self.epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    def save_network(self, ep):
        save_path = 'checkpoint/ac_ep_{}.h5'
        self.model.save_weights(save_path.format(ep))


if __name__ == '__main__':
    model = ActorCritic(8, 16, 9)

    acts, critic = model.forward(tf.convert_to_tensor(np.random.rand(1, 16)))
    print(critic)
    # acts_2 = model.act(acts)
    # print(acts_2)
    # print(acts[0][0, acts_2[0]])
