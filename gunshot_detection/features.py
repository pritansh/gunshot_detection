import numpy as np
from librosa import load, feature

class Features:

    def __init__(self, files):
        self.classes = len(files)
        self.features_dim = 20
        total_files = [len(e) for e in files]
        self.features = np.empty((0, self.features_dim))
        self.labels = np.zeros((sum(total_files), self.classes))
        index = 0
        for i in range(0, self.classes):
            for j in range(0, total_files[i]):
                samples, rate = np.array(load(files[i][j]))
                self.features = np.vstack([
                    self.features, np.mean(feature.mfcc(y=samples, sr=rate).T, axis=0)])
            self.labels[index:index+total_files[i], i] = 1
            index = total_files[i]
        self.features = np.array(self.features)
        self.labels = np.array(self.labels, dtype='int')
        print np.shape(self.features), np.shape(self.labels)

    def __str__(self):
        feature_str = 'Feature Vector for ' + str(self.classes) + ' Classes'
        feature_str += '\n\tFeatures -> ' + str(np.shape(self.features)[0])
        feature_str += '\n\tDimensions -> ' + str(self.features_dim)
        return feature_str

    def __repr__(self):
        feature_str = 'Feature Vector for ' + str(self.classes) + ' Classes'
        feature_str += '\n\tFeatures -> ' + str(np.shape(self.features)[0])
        feature_str += '\n\tDimensions -> ' + str(self.features_dim)
        return feature_str
