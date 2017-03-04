import numpy as np
from librosa import load, feature

from ml.progress import print_progress

def ind2vec(ind, N=None):
    ''''''
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return (np.arange(N) == ind[:, None]).astype(int)

class AudioFeatures:
    ''''''
    def __init__(self, files, feature_reduction='', dims=20, vector_reduction='mean'):
        self.classes = len(files)
        self.features_dim = dims
        total_files = [len(e) for e in files]
        done = 0
        total = sum(total_files)
        self.features = np.empty((0, self.features_dim))
        self.labels = np.zeros((0, self.classes))
        print 'Extracting Features ->'
        for i in range(0, self.classes):
            for j in range(0, total_files[i]):
                samples, rate = np.array(load(files[i][j]))
                mfcc = feature.mfcc(y=samples, sr=rate).T
                if vector_reduction == 'mean':
                    mfcc = np.mean(mfcc, axis=0)
                elif vector_reduction == 'iqr':
                    mfcc = np.percentile(mfcc, 50, axis=0)
                elif vector_reduction == 'var':
                    mfcc = np.var(mfcc, axis=0)
                self.features = np.vstack([
                    self.features, mfcc])
                if mfcc.ndim > 1:
                    self.labels = np.vstack([
                        self.labels, ind2vec([i for e in range(0, len(mfcc[0]))], N=self.classes)])
                else:
                    self.labels = np.vstack([self.labels, ind2vec([i], N=self.classes)])
                done += 1
                print_progress(iteration=done, total=total)
        self.features = np.array(self.features)
        self.labels = np.array(self.labels, dtype='int')

    def __str__(self):
        feature_str = 'Audio Feature Vector for ' + str(self.classes) + ' Classes'
        feature_str += '\n\tFeatures -> ' + str(np.shape(self.features)[0])
        feature_str += '\n\tDimensions -> ' + str(self.features_dim)
        return feature_str

    def __repr__(self):
        feature_str = 'Audio Feature Vector for ' + str(self.classes) + ' Classes'
        feature_str += '\n\tFeatures -> ' + str(np.shape(self.features)[0])
        feature_str += '\n\tDimensions -> ' + str(self.features_dim)
        return feature_str
