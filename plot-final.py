import numpy as np
import matplotlib.pyplot as plt

acc = np.load('./acc.npy')
acc = np.reshape(acc, (4, 3))

n_groups = 3
fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.18

opacity = 0.4
error_config = {'ecolor': '0.3'}

r1 = plt.bar(index, tuple(acc[0]), bar_width, alpha=opacity, color='b', label='Chroma')
r2 = plt.bar(index + bar_width, tuple(acc[1]), bar_width, alpha=opacity, color='r', label='Perceptual')
r3 = plt.bar(index + 2 * bar_width, tuple(acc[2]), bar_width, alpha=opacity, color='g', label='Spectral')
r4 = plt.bar(index + 3 * bar_width, tuple(acc[3]), bar_width, alpha=opacity, color='y', label='Others')

plt.xlabel('Feature Reduction')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Feature Reduction Algorithm Plot')
plt.xticks(index + 3 * bar_width / 2, ('No', 'PCA', 'LDA'))
plt.legend()

plt.tight_layout()
plt.show()