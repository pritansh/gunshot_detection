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
    all_comb.extend([list(x) for x in combinations(dirs, i)])


# Part 3 - Training
accuracy = []
for i in range(0, 1):
    print 'Network training of ' + str(i+1) + 'dataset -> ' + str(all_comb[i])
    train = ml.AudioFeatures(filename=feature_dir + '/train', class_dirs=all_comb[0])
    test = ml.AudioFeatures(filename=feature_dir + '/test', class_dirs=all_comb[0])
    net = ml.MLP(feature_dim=train.features_dim, classes=train.classes, hidden_units=[300, 500])
    print test.labels
    accuracy.append(net.train(train=train, test=test, epochs=500))

print accuracy