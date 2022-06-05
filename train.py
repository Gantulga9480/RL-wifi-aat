from environment import Environment
from model import DQN, ReplayBuffer

MAX_BUFFER = 100000
MIN_BUFFER = 5000
TARGET_SET_FREQ = 5

env = Environment()
model = DQN()
replay_buffer = ReplayBuffer(MAX_BUFFER, MIN_BUFFER)

try:
    while True:
        state = env.reset()
        while not env.over:
            action = model.predict_action(state)
            reward, n_state = env.step(action)
            replay_buffer.push([state, action, n_state, reward, env.over])
            state = n_state
            env.plot()

            if replay_buffer.trainable:
                model.train(replay_buffer.sample(model.BATCH_SIZE))
                model.decay_epsilon()

            # if not env.episode_count % TARGET_SET_FREQ:

            print('epsilon:', model.epsilon)

        if replay_buffer.trainable:
            model.update_target()
            model.save(f'models/model_{env.episode_count}_{round(env.episode_r, 3)}')
        env.save()
except KeyboardInterrupt():
    ...
