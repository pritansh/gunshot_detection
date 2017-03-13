import numpy as np
from librosa import load, feature
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from ml.progress import print_progress

def ind2vec(ind, N=None):
    ''''''
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return (np.arange(N) == ind[:, None]).astype(int)

class AudioFeatures:
    ''''''
    def __init__(self, files, class_dirs=[], features=['mfcc'], filename='',
                 mfcc_coeff=20, chroma_coeff=12, poly_order=1,
                 feature_reduction='', reduction_size=10, vector_reduction='mean'):
        if len(class_dirs) > 0:
            self.load_classes(filename, class_dirs)
            return
        if len(filename) > 0:
            self.load(filename)
            return
        self.feature_functions = {
            'chroma_stft': [
                dict(y=None, sr=None, n_chroma=None), feature.chroma_stft, chroma_coeff],
            #'chroma_cqt': feature.chroma_cqt,
            #'chrome_cens': feature.chroma_cens,
            'melspectrogram': [dict(y=None, sr=None), feature.melspectrogram, 128],
            'mfcc': [dict(y=None, sr=None, n_mfcc=None), feature.mfcc, mfcc_coeff],
            'rmse': [dict(y=None), feature.rmse, 1],
            'spectral_bandwidth': [dict(y=None, sr=None), feature.spectral_bandwidth, 1],
            'spectral_centroid': [dict(y=None, sr=None), feature.spectral_centroid, 1],
            'spectral_contrast': [dict(y=None, sr=None), feature.spectral_contrast, 1],
            'spectral_rolloff': [dict(y=None, sr=None), feature.spectral_rolloff, 1],
            'poly_features': [
                dict(y=None, sr=None, order=None), feature.poly_features, poly_order+1],
            #'tonnetz': feature.tonnetz,
            'zcr': [dict(y=None), feature.zero_crossing_rate, 1]
        }
        self.files = []
        self.classes = len(files)
        self.features_info = []
        self.features_dim = 0
        for ftr in features:
            self.features_dim += self.feature_functions[ftr][2]
            self.features_info.append((ftr, self.feature_functions[ftr][2]))
        self.total_files = np.array([len(e) for e in files])
        done = 0
        total = np.sum(self.total_files)
        self.features = np.empty((0, self.features_dim))
        self.labels = np.zeros((0, self.classes))
        print 'Extracting Features ->'
        for i in range(0, self.classes):
            for j in range(0, self.total_files[i]):
                samples, rate = np.array(load(files[i][j]))
                params = dict(y=samples, sr=rate, n_chroma=chroma_coeff,
                              n_mfcc=mfcc_coeff, order=poly_order)
                vector = np.empty(0)
                for ftr in features:
                    fxn = self.feature_functions[ftr]
                    fxn[0] = {k: params[k] for k, v in fxn[0].iteritems()}
                    if np.shape(vector)[0] == 0:
                        vector = fxn[1](**fxn[0])
                    else:
                        vector = np.vstack([vector, fxn[1](**fxn[0])])
                vector = vector.T
                if vector_reduction == 'mean':
                    vector = np.mean(vector, axis=0)
                elif vector_reduction == 'iqr':
                    vector = np.percentile(vector, 50, axis=0)
                elif vector_reduction == 'var':
                    vector = np.var(vector, axis=0)
                self.features = np.vstack([
                    self.features, vector])
                if vector.ndim > 1:
                    self.labels = np.vstack([
                        self.labels, ind2vec(
                            [i for e in range(0, len(vector[0]))], N=self.classes)])
                else:
                    self.labels = np.vstack([self.labels, ind2vec([i], N=self.classes)])
                done += 1
                print_progress(iteration=done, total=total)
        self.features = np.array(self.features)
        self.labels = np.array(self.labels, dtype='int')
        if feature_reduction == 'pca':
            pca_clf = PCA(n_components=reduction_size).fit(self.features)
            self.features = pca_clf.transform(self.features)

    def save(self, filename=''):
        np.save(filename + '-features.npy', self.features)
        np.save(filename + '-labels.npy', self.labels)

    def save_classes(self, filename='', class_dirs=[]):
        csum = np.cumsum(self.total_files)
        for i in range(0, self.classes):
            if i == 0:
                min_row, max_row = (0, csum[i])
            elif i == self.classes - 1:
                min_row, max_row = (csum[i-1], len(self.features))
            else:
                min_row, max_row = (csum[i-1], csum[i])
            name = filename + '-' + class_dirs[i]
            np.save(name + '-features.npy', self.features[min_row:max_row][:])
            np.save(name + '-labels.npy', self.labels[min_row:max_row][:])

    def load(self, filename):
        self.features = np.load(filename + '-features.npy')
        self.labels = np.load(filename + '-labels.npy')
        self.features_dim = len(self.features[0])
        self.classes = len(self.labels[0])

    def load_classes(self, filename, class_dirs):
        name = filename + '-' + class_dirs[0]
        self.features = np.load(name + '-features.npy')
        self.labels = np.load(name + '-labels.npy')
        for i in range(1, len(class_dirs)):
            name = filename + '-' + class_dirs[i]
            self.features = np.vstack([self.features, np.load(name + '-features.npy')])
            self.labels = np.vstack([self.labels, np.load(name + '-labels.npy')])
        self.features_dim = len(self.features[0])
        self.classes = len(self.labels[0])

    def __str__(self):
        feature_str = 'Audio Feature Vector for ' + str(self.classes) + ' Classes'
        feature_str += '\n\tFeatures -> ' + str(np.shape(self.features)[0])
        feature_str += '\n\tDimensions -> ' + str(self.features_dim)
        feature_str += '\n\tInfo -> ' + str(self.features_info)
        return feature_str

    def __repr__(self):
        feature_str = 'Audio Feature Vector for ' + str(self.classes) + ' Classes'
        feature_str += '\n\tFeatures -> ' + str(np.shape(self.features)[0])
        feature_str += '\n\tDimensions -> ' + str(self.features_dim)
        feature_str += '\n\tInfo -> ' + str(self.features_info)
        return feature_str
