from sklearn.svm import SVC, NuSVC
from sklearn.metrics import confusion_matrix
import numpy as np

from ml.features import AudioFeatures
from ml.progress import print_progress

class SVM:
    ''''''
    def __init__(self, features_dim, classes, svm_type='nu' ,kernel='linear', poly_degree=3):
        ''''''
        self.kernel = kernel
        self.poly_degree = poly_degree
        self.svm_type = svm_type
        self.svms = {
            'nu': NuSVC(kernel=self.kernel, degree=self.poly_degree),
            'c': SVC(kernel=self.kernel, degree=self.poly_degree)
        }
        self.svm = self.svms[svm_type]
        self.features_dim = features_dim
        self.classes = classes

    def train(self, train, test):
        ''''''
        self.svm.fit(train.features, train.class_labels)
        pred = self.svm.predict(test.features)
        cfm = confusion_matrix(test.class_labels, pred)
        true = 0
        size = np.shape(cfm)[0]
        total = float(np.sum(np.sum(cfm, axis=0)))
        for i in range(0, size):
            true += cfm[i][i]
        test_accuracy = true/total
        print 'Test accuracy: ', test_accuracy
        return test_accuracy

    def __str__(self):
        svm_str = 'SVM Config'
        svm_str += '\n\tType -> ' + self.svm_type
        svm_str += '\n\tKernel -> ' + self.kernel
        if self.kernel == 'poly':
            svm_str += ' , Degree -> ' + str(self.poly_degree)
        svm_str += '\n\tInput -> ' + str(self.features_dim)
        svm_str += '\n\tOutput -> ' + str(self.classes)
        return svm_str

    def __repr__(self):
        return str(self)
