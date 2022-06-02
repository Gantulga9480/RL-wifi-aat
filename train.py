from environment import Environment
from model import DQN, ReplayBuffer

MAX_BUFFER = 10000
MIN_BUFFER = 1000
TARGET_SET_FREQ = 5

env = Environment()
model = DQN([
    (25, 'relu'),
    (25, 'relu'),
    (4, 'linear')
])
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

            if not env.episode_count % TARGET_SET_FREQ:
                model.update_target()

            print('epsilon:', model.epsilon)

        model.save_model()
        env.save()
except KeyboardInterrupt():
    ...