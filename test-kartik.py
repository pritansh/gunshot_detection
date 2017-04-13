import ml
import numpy as np

dir_path = './final/kartik/'

train = ml.AudioFeatures().extract(filename=dir_path + 'final-train')
test = ml.AudioFeatures().extract(filename=dir_path + 'final-test')

def labels_to_class(labels):
    temp = []
    for i in range(0, len(labels)):
        if labels[i][0] == 1:
            temp.append(0)
        elif labels[i][1] == 1:
            temp.append(1)
    return temp

train.class_labels = np.array(labels_to_class(train.labels))
test.class_labels = np.array(labels_to_class(test.labels))

ncd = dict(features_dim=21, classes=2, hidden_units=[80, 100], learn_rate=0.01)
nc = ml.NetworkConfig(**ncd)

net = ml.MLP(nc)

print net

net.train(train=train, test=test, epochs=5000)

svm = ml.SVM(features_dim=21, classes=2, svm_type='c', kernel='rbf', poly_degree=3)
print svm

svm.train(train=train, test=test)
