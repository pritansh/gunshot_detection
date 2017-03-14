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

class FeaturesConfig:
    ''''''
    def __init__(self, features, mfcc_coeff=20, chroma_coeff=12, poly_order=1):
        self.features = features
        self.mfcc_coeff = mfcc_coeff
        self.chroma_coeff = chroma_coeff
        self.poly_order = poly_order

    def __str__(self):
        f_cfg_str = 'Features Config ->'
        f_cfg_str += '\n\tFeatures -> ' + str(self.features)
        f_cfg_str += '\n\tCoeff -> ' + str(
            (('MFCC', self.mfcc_coeff), ('Chroma', self.chroma_coeff),
             ('Poly order', self.poly_order)))
        return f_cfg_str

    def __repr__(self):
        return str(self)


class ReductionConfig:
    ''''''
    def __init__(self, feature_reduction='', reduction_size=10, vector_reduction='mean'):
        self.feature_reduction = feature_reduction
        self.reduction_size = reduction_size
        self.vector_reduction = vector_reduction

    def __str__(self):
        r_cfg_str = 'Reduction Config ->'
        r_cfg_str += '\n\tFeature Reduction -> ' + self.feature_reduction
        r_cfg_str += '\n\tReduction size -> ' + str(self.reduction_size)
        r_cfg_str += '\n\tVector Reduction -> ' + self.vector_reduction
        return r_cfg_str

    def __repr__(self):
        return str(self)

class Info:
    ''''''
    def __init__(self, files, features_config, reduction_config, info):
        self.files = files
        self.features_config = features_config
        self.reduction_config = reduction_config
        self.info = info
        self.total_files = np.array([len(e) for e in self.files])


class AudioFeatures:
    ''''''
    def __init__(self, files,
                 features_cfg=FeaturesConfig(
                     features=['mfcc'], mfcc_coeff=20, chroma_coeff=12, poly_order=1),
                 reduction_cfg=ReductionConfig(
                     feature_reduction='', reduction_size=10, vector_reduction='mean')):
        self.functions = {
            'chroma_stft': [
                dict(y=None, sr=None, n_chroma=None),
                feature.chroma_stft, features_cfg.chroma_coeff],
            #'chroma_cqt': feature.chroma_cqt,
            #'chrome_cens': feature.chroma_cens,
            'melspectrogram': [dict(y=None, sr=None), feature.melspectrogram, 128],
            'mfcc': [dict(y=None, sr=None, n_mfcc=None), feature.mfcc, features_cfg.mfcc_coeff],
            'rmse': [dict(y=None), feature.rmse, 1],
            'spectral_bandwidth': [dict(y=None, sr=None), feature.spectral_bandwidth, 1],
            'spectral_centroid': [dict(y=None, sr=None), feature.spectral_centroid, 1],
            'spectral_contrast': [dict(y=None, sr=None), feature.spectral_contrast, 1],
            'spectral_rolloff': [dict(y=None, sr=None), feature.spectral_rolloff, 1],
            'poly_features': [
                dict(y=None, sr=None, order=None),
                feature.poly_features, features_cfg.poly_order+1],
            #'tonnetz': feature.tonnetz,
            'zcr': [dict(y=None), feature.zero_crossing_rate, 1],
            'mean': [[None, 0], np.mean],
            'iqr': [[None, 50, 0], np.percentile],
            'var': [[None, 0], np.var]
        }
        self.data_info = Info(
            files=files, features_config=features_cfg, reduction_config=reduction_cfg, info=[])
        self.classes = len(files)
        self.features_dim = 0
        for ftr in features_cfg.features:
            self.features_dim += self.functions[ftr][2]
            self.data_info.info.append((ftr, self.functions[ftr][2]))
        self.features = np.empty((0, self.features_dim))
        self.labels = np.zeros((0, self.classes))

    def extract(self, class_dirs=None, filename=None):
        ''''''
        if class_dirs != None:
            self.load_classes(filename, class_dirs)
            return
        if filename != None:
            self.load(filename)
            return
        done = 0
        total = np.sum(self.data_info.total_files)
        print 'Extracting Features ->'
        for i in range(0, self.classes):
            for j in range(0, self.data_info.total_files[i]):
                samples, rate = np.array(load(self.data_info.files[i][j]))
                params = dict(y=samples, sr=rate,
                              n_chroma=self.data_info.features_config.chroma_coeff,
                              n_mfcc=self.data_info.features_config.mfcc_coeff,
                              order=self.data_info.features_config.poly_order)
                vector = np.empty(0)
                for ftr in self.data_info.features_config.features:
                    fxn = self.functions[ftr]
                    fxn[0] = {k: params[k] for k, v in fxn[0].iteritems()}
                    if np.shape(vector)[0] == 0:
                        vector = fxn[1](**fxn[0])
                    else:
                        vector = np.vstack([vector, fxn[1](**fxn[0])])
                vector = vector.T
                reduction = self.functions[self.data_info.reduction_config.vector_reduction]
                reduction[0][0] = vector
                vector = reduction[1](*reduction[0])
                self.features = np.vstack([
                    self.features, vector])
                if vector.ndim > 1:
                    self.labels = np.vstack([
                        self.labels, ind2vec(
                            np.repeat(i, len(vector[0])), N=self.classes)])
                else:
                    self.labels = np.vstack([self.labels, ind2vec([i], N=self.classes)])
                done += 1
                print_progress(iteration=done, total=total)
        self.features = np.array(self.features)
        self.labels = np.array(self.labels, dtype='int')
        if self.data_info.reduction_config.feature_reduction == 'pca':
            pca_clf = PCA(
                n_components=self.data_info.reduction_config.reduction_size).fit(self.features)
            self.features = pca_clf.transform(self.features)
        return self

    def save(self, filename=''):
        np.save(filename + '-features.npy', self.features)
        np.save(filename + '-labels.npy', self.labels)

    def save_classes(self, filename='', class_dirs=[]):
        csum = np.cumsum(self.data_info.total_files)
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
        feature_str += '\n\tVectors -> ' + str(np.shape(self.features)[0])
        feature_str += '\n\tFeatures -> ' + str(self.features_dim)
        feature_str += '\n\tInfo -> ' + str(self.data_info.info)
        feature_str += '\n\t' + str(self.data_info.features_config)
        feature_str += '\n\t' + str(self.data_info.reduction_config)
        return feature_str

    def __repr__(self):
        return str(self)
