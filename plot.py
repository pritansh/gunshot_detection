import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np

fig = plt.figure(figsize=(20, 20))
g = gs.GridSpec(5, 2)

fig.set_xlabel('X')
fig.set_ylabel('Y')

data = np.empty((0, 20))
for i in range(0, 10):
    data = np.vstack([data, np.random.randn(20)])

plots = []

for i in range(0, 10):
    plots.append(fig.add_subplot(5, 2, i+1))
    plots[i].set_xlim([0, 20])
    plots[i].set_ylim([-2, 2])
    plots[i].plot(data[:][i])

plt.show()