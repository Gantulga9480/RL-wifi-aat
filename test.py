from environment import Environment
import random


MAX_EPISODE = 2

env = Environment()
episode = 0

while True:

    state = env.reset()

    while not env.over:

        r = env.step(random.randint(0, 3))

        print(r)

    episode += 1
    if episode == MAX_EPISODE:
        exit()
