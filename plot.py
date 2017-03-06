import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(nrows=6, ncols=2, figsize=(6, 6), sharex=True, sharey=True)

data = np.load('./features/final-accuracy.npy')
data = np.reshape(data, (10, -1, 7))

labels = ['Mean', 'Mean PCA', 'Var', 'Var PCA', 'IQR', 'IQR PCA']
lines = []
layer = 0

for i in range(0, 6):
    for j in range(0, 2):
        if i == 0:
            ax[i, j].set_frame_on(False)
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
        else:
            ax[i, j].set_xlim([1, 6])
            ax[i, j].set_ylim([0.5, 1.0])
            title = str(layer+1) + ' Hidden Layer'
            if layer+1 > 1:
                title += 's'
            layer += 1
            ax[i, j].set_title(title)
            for k in range(0, 6):
                temp = ax[i, j].plot(data[:][i][k], label=labels[k])
                if i == 5:
                    lines.append(temp[0])

fig.text(0.5, 0.005, 'Number of neurons * 50', horizontalalignment='center')
fig.text(0.005, 0.5, 'Accuracy', verticalalignment='center', rotation='vertical')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.figlegend(lines, labels,'upper center', ncol=3, bbox_to_anchor=(0.5, 0.95))
plt.show()