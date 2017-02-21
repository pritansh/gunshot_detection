import numpy as np
from librosa import load, feature

def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None: 
        N = ind.max() + 1
    return (np.arange(N) == ind[:,None]).astype(int)

class Features:

    def __init__(self, files):
        self.classes = len(files)
        self.features_dim = 20
        total_files = [len(e) for e in files]
        self.features = np.empty((0, self.features_dim))
        self.labels = np.zeros((0, self.classes))
        for i in range(0, self.classes):
            for j in range(0, total_files[i]):
                samples, rate = np.array(load(files[i][j]))
                self.features = np.vstack([
                    self.features, np.mean(feature.mfcc(y=samples, sr=rate).T, axis=0)])
            self.labels = np.vstack([
                self.labels, ind2vec([i for e in range(0, total_files[i])], N=self.classes)])
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
