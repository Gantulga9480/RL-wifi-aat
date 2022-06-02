from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
import numpy as np
import random
from collections import deque
from matplotlib import pyplot as plt
from datetime import datetime as dt
from environment import Environment

# Hyperparameters
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.9
EPSILON = 1
EPSILON_DECAY = 0.99999
MIN_EPSILON = 0.01
BATCH_SIZE = 256
EPOCH = 5
TARGET_NET_UPDATE_FREQUENCY = 5
UPDATE_COUNTER = 1
MAX_SCORE = 5

BUFFER_SIZE = 20000
MIN_BUFFER_SIZE = 5000
REPLAY_BUFFER = deque(maxlen=BUFFER_SIZE)
SAMPLES = []

# Mixed precision : RTX GPU only
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# CPU only
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


show_every = 5
episode = 1
history = {'ep': [], 'reward': []}
ep_rewards = []
reward_tmp = 0
hour = 1
ep_index = 1


# Create new model with given specs
def get_model():
    model = Sequential()
    model.add(Input(shape=(107,), name='input'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(4, activation='linear'))
    # model.add(Activation('linear', dtype='float32'))  # TC enabled GPU only
    model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE),
                  metrics=["accuracy"])
    model.summary()
    return model


# Fit model
def keras_train():
    SAMPLES = random.sample(REPLAY_BUFFER, BATCH_SIZE)
    current_states = np.array([item[0] for item in SAMPLES])
    new_current_state = np.array([item[2] for item in SAMPLES])
    current_qs_list = []
    future_qs_list = []
    current_qs_list = main_nn.predict(current_states)
    future_qs_list = target_nn.predict(new_current_state)

    X = []
    Y = []
    for index, (state, action, _, reward, done) in enumerate(SAMPLES):
        if not done:
            new_q = reward + DISCOUNT_RATE * np.max(future_qs_list[index])
        else:
            new_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = new_q

        X.append(state)
        Y.append(current_qs)
    main_nn.fit(np.array(X), np.array(Y), epochs=EPOCH,
                batch_size=BATCH_SIZE, shuffle=False, verbose=1)


# Load model
main_nn = get_model()
target_nn = get_model()
target_nn.set_weights(main_nn.get_weights())
start_time = dt.now()

game = Environment()

while True:
    state = game.reset()
    ep_reward = 0
    score = 0
    while not game.over:
        if np.random.random() < EPSILON:
            action = np.random.randint(3)
        else:
            action_values = main_nn.predict(np.expand_dims(state, axis=0))[0]
            action = np.argmax(action_values)
        terminal, new_state, r = game.step(action=action)
        if r == FOOD_REWARD:
            score += 1
            REPLAY_BUFFER.append([state, action, new_state, r, True])
        else:
            REPLAY_BUFFER.append([state, action, new_state, r, terminal])
        EPSILON = EPSILON * EPSILON_DECAY
        ep_reward += r
        state = new_state

        if len(REPLAY_BUFFER) > MIN_BUFFER_SIZE:
            keras_train()
            UPDATE_COUNTER += 1

        if UPDATE_COUNTER % TARGET_NET_UPDATE_FREQUENCY == 0:
            target_nn.set_weights(main_nn.get_weights())
            UPDATE_COUNTER = 1
        if EPSILON < MIN_EPSILON:
            EPSILON = 0.1
            info = f"{args.index}_sc{MAX_SCORE}_ep{episode}_t{dtime[0]}.h5"
            main_nn.save(info)
            ep_index += 1

    ep_rewards.append(ep_reward)
    episode += 1

    if score > MAX_SCORE:
        MAX_SCORE = score
        time_now = dt.now()
        dtime = str(time_now - start_time).split(':')
        info = f"{args.index}_sc{MAX_SCORE}_ep{episode}_t{dtime[0]}.h5"
        main_nn.save(info)

    if episode % show_every == 0:
        time_now = dt.now()
        dtime = str(time_now - start_time).split(':')
        if int(dtime[0]) == hour:
            hour += 1
        avg_r = sum(ep_rewards) / show_every
        ep_rewards.clear()
        if avg_r > reward_tmp:
            desc = f'avg: ↑ {avg_r} ep: {episode} eps: {EPSILON} '
        else:
            desc = f'avg: ↓ {avg_r} ep: {episode} eps: {EPSILON} '
        desc += f'{dtime[0]}:{dtime[1]}'
        reward_tmp = avg_r
        history['reward'].append(avg_r)
        history['ep'].append(EPSILON*10)
        game.caption(desc)
