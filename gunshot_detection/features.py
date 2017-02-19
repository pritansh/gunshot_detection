import numpy as np
from librosa import load, feature

class Features:

    def __init__(self, files):
        self.classes = len(files)
        self.features_dim = 20
        total_files = [len(e) for e in files]
        self.features = np.empty(sum(total_files), dtype='object')
        self.labels = np.empty(sum(total_files), dtype='int8')
        counter = 0
        for i in range(0, self.classes):
            for j in range(0, total_files[i]):
                samples, rate = np.array(load(files[i][j]))
                self.features[counter+j] = np.mean(feature.mfcc(y=samples, sr=rate).T, axis=0)
            self.labels[:total_files[i]] = i+1
            counter += total_files[i]

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
