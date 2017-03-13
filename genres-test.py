from itertools import combinations
import numpy as np
import ml
import os

feature_dir = './genres/data'

'''
# Part 1 - Feature
d = ml.Dataset('./genres/wav')
train = ml.AudioFeatures(files=d.train_files, vector_reduction='mean')
train.save_classes(filename=feature_dir + '/train', class_dirs=d.class_dirs)
test = ml.AudioFeatures(files=d.test_files, vector_reduction='mean')
test.save_classes(filename=feature_dir + '/test', class_dirs=d.class_dirs)
'''

# Part 2 - Combinations
dirs = os.listdir('./genres/wav')
min_dirs, max_dirs = (2, 9)
all_comb = []

for i in range(min_dirs, max_dirs):
    all_comb.append([list(x) for x in combinations(dirs, i)])


# Part 3 - Training
accuracy = []
upto = 5
for i in range(upto-1, upto):
    print 'Classes -> ' + str(i+2) + ' -> ' + str(len(all_comb[i]))
    for j in range(0, len(all_comb[i])):
        print 'Network training of ' + str(j+1) + ' dataset -> ' + str(all_comb[i][j])
        train = ml.AudioFeatures(filename=feature_dir + '/train', class_dirs=all_comb[i][j])
        test = ml.AudioFeatures(filename=feature_dir + '/test', class_dirs=all_comb[i][j])
        net = ml.MLP(feature_dim=train.features_dim, classes=len(all_comb[i][j]), hidden_units=[300, 500])
        train.labels = train.labels[:, ~(train.labels==0).all(0)]
        test.labels = test.labels[:, ~(test.labels==0).all(0)]
        accuracy.append(net.train(train=train, test=test, epochs=500))

print accuracy
np.save('./genres/data/accuracy' + str(upto-1) + '.npy', np.array(accuracy))
np.save('./genres/data/combinations' + str(upto-1) + '.npy', np.array(all_comb))
max_accuracy = max(accuracy)
index = accuracy.index(max_accuracy)
max_comb = all_comb[upto-1][index]
max_acc = accuracy[upto-1][index]

print max_accuracy, index
print max_acc, max_comb