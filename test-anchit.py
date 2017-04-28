import ml
import numpy as np

class Features:
    def __init__(self, filename):
        self.features = np.load(filename + '-features.npy')
        self.class_labels = np.load(filename + '-labels.npy')

train = Features(filename='./anchit/train')
test = Features(filename='./anchit/test')

svm = ml.SVM(features_dim=13, classes=3, svm_type='c', kernel='poly', poly_degree=3)
print svm

svm.train(train=train, test=test)