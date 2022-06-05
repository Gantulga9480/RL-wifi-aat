import matplotlib.pyplot as plt


lines = []

with open('6_4/reward_history2.txt', 'r') as f:
    for line in f.readlines():
        lines.append(float(line.rstrip()))

plt.plot(lines)
plt.show()
