import socket
from matplotlib import pyplot as plt

HOST = "127.0.0.1"
PORT = 8888


class Environment:

    MAX_STEP = 107

    def __init__(self) -> None:
        self.soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.soc.connect((HOST, PORT))
        print("Connected to simulation server")

        self.over = False
        self.step_count = 0
        self.episode_count = 0
        self.episode_r = 0
        self.reward_hist = []

        self.graph = plt.subplot()

        self.state = []

    def __del__(self):
        self.soc.close()
        print("Connectoion closed")

    def step(self, action):
        self.state[self.step_count] = action
        req_data = f'{self.step_count},' + ','.join([str(s) for s in self.state]) + '\n'
        self.soc.sendall(req_data.encode())
        res = self.soc.recv(100).decode().rstrip().split(',')
        if int(res[0]) != self.step_count:
            print("Env step Error")
            return
        self.step_count += 1
        self.episode_r += float(res[1])
        if self.step_count == self.MAX_STEP:
            self.over = True
            self.episode_count += 1
            self.reward_hist.append(self.episode_r)
        st = [self.step_count]
        st.extend(self.state)
        return float(res[1]), st

    def reset(self):
        print('Environment reset')
        self.over = False
        self.state = []
        self.step_count = 0
        self.episode_r = 0
        for _ in range(self.MAX_STEP):
            self.state.append(0)
        st = [self.step_count]
        st.extend(self.state)
        self.graph.cla()
        self.graph.plot(self.reward_hist)
        return st

    def save(self):
        with open('reward_history.txt', 'w') as f:
            for val in self.reward_hist:
                f.write(str(val)+'\n')

    def plot(self):
        plt.pause(0.00001)
