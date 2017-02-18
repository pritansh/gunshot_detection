import numpy as np
from librosa import load, feature

class Features:

    def __init__(self, files):
        classes = len(files)
        itr = 0
        self.features = np.empty(classes, dtype='object')
        self.labels = np.empty(classes, dtype='object')
        for i in range(0, classes):
            total_files = len(files[i])
            self.features[i] = np.empty((total_files, 20))
            self.labels = np.empty(total_files, dtype='int')
            for j in range(0, total_files):
                samples, rate = np.array(load(files[i][j]))
                self.features[i][itr+j] = np.mean(feature.mfcc(y=samples, sr=rate).T, axis=0)
            self.labels[i][itr:itr+total_files] = i+1
            itr += total_files
        