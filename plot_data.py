import matplotlib.pyplot as plt


lines = []

with open('reward_history.txt', 'r') as f:
    for line in f.readlines():
        lines.append(float(line.rstrip()))

plt.plot(lines)
plt.show()
