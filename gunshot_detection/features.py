import numpy as np
from librosa import load, feature

class Features:

    def __init__(self, files):
        classes = len(files)
        self.features = np.empty(classes, dtype='object')
        self.labels = np.empty(classes, dtype='object')
        for i in range(0, classes):
            total_files = len(files[i])
            self.features[i] = np.empty((total_files, 20))
            self.labels[i] = np.empty(total_files, dtype='int8')
            for j in range(0, total_files):
                samples, rate = np.array(load(files[i][j]))
                self.features[i][j] = np.mean(feature.mfcc(y=samples, sr=rate).T, axis=0)
            self.labels[i][:total_files] = i+1

    def __str__(self):
        n_classes = np.shape(self.features)[0]
        feature_str = 'Feature Vector for ' + str(n_classes) + ' Classes'
        for i in range(0, n_classes):
            dim = np.shape(self.features[i])
            feature_str += '\n Class ' + str(i+1) + ' -> '
            feature_str += str(dim[0]) + ' files & ' + str(dim[1]) + ' features'
        return feature_str

    def __repr__(self):
        n_classes = np.shape(self.features)[0]
        feature_str = 'Feature Vector for ' + str(n_classes) + 'Classes'
        for i in range(0, n_classes):
            dim = np.shape(self.features[i])
            feature_str += '\n Class ' + str(i+1) + ' -> '
            feature_str += str(dim[0]) + ' files & ' + str(dim[1]) + ' features'
        return feature_str
