import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Activation, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from collections import deque
import numpy as np
import random


class DQN:

    LEARNING_RATE = 0.001
    DISCOUNT_RATE = 0.999
    BATCH_SIZE = 512
    EPOCHS = 5
    EPSILON_DECAY = 0.99995
    MIN_EPSILON = 0.1

    def __init__(self) -> None:

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.list_logical_devices('GPU')
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.main_model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.main_model.get_weights())

        self.epsilon = 1

    def create_model(self) -> Sequential:
        model = Sequential()
        model.add(Input(shape=(108,)))
        model.add(Dropout(0.3))
        model.add(Dense(54, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(27, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss="mse",
                      optimizer=Adam(learning_rate=self.LEARNING_RATE),
                      metrics=["accuracy"])
        model.summary()
        return model

    def save(self, name='model.h5'):
        if not name.endswith('.h5'):
            name += '.h5'
        self.main_model.save(name)

    def load(self, path='model.h5'):
        if not path.endswith('.h5'):
            path += '.h5'
        try:
            self.main_model = load_model(path)
            self.target_model.set_weights(self.main_model.get_weights())
        except IOError:
            print('Model file not found!')
            exit()

    def update_target(self):
        self.target_model.set_weights(self.main_model.get_weights())

    def predict_action(self, state):
        if np.random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            action_values = self.main_model.predict(np.expand_dims(state, axis=0))[0]
            return np.argmax(action_values)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.EPSILON_DECAY, self.MIN_EPSILON)

    def train(self, samples):
        current_states = np.array([item[0] for item in samples])
        new_current_state = np.array([item[2] for item in samples])
        current_qs_list = []
        future_qs_list = []
        current_qs_list = self.main_model.predict(current_states)
        future_qs_list = self.target_model.predict(new_current_state)

        X = []
        Y = []
        for index, (state, action, _, reward, done) in enumerate(samples):
            if not done:
                new_q = reward + self.DISCOUNT_RATE * np.max(future_qs_list[index])
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(state)
            Y.append(current_qs)
        self.main_model.fit(np.array(X), np.array(Y),
                            epochs=self.EPOCHS,
                            batch_size=self.BATCH_SIZE,
                            shuffle=False,
                            verbose=2)


class ReplayBuffer:

    def __init__(self, max_size, min_size) -> None:
        self.max_size = max_size
        self.min_size = min_size
        self.buffer = deque(maxlen=max_size)

    @property
    def trainable(self):
        return self.buffer.__len__() > self.min_size

    def push(self, data):
        self.buffer.append(data)

    def sample(self, sample_size):
        return random.sample(self.buffer, sample_size)
