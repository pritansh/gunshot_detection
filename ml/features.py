import numpy as np
import copy
from librosa import load, feature
from sklearn.decomposition import PCA, IncrementalPCA as IPCA, KernelPCA as KPCA
from sklearn.decomposition import FastICA as FICA, TruncatedSVD as TSVD, NMF, SparsePCA as SPCA
from sklearn.cross_decomposition import CCA, PLSSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from ml.progress import print_progress

def ind2vec(ind, classes=None):
    '''
        Function to convert indices to matrix form.
        ind = indices -> list
        classes = number of classes -> int
        Example :->
            indices = [0, 1, 2, 3] , classes = 4
            returns matrix form = [[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]]
    '''
    ind = np.asarray(ind)
    if classes is None:
        classes = ind.max() + 1
    return (np.arange(classes) == ind[:, None]).astype(int)

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
        coeff_str = ''
        if 'mfcc' in self.features:
            coeff_str += str(('MFCC', self.mfcc_coeff))
        if 'c_stft' in self.features or 'c_cqt' in self.features or 'c_cens' in self.features:
            if len(coeff_str) != 0:
                coeff_str += ', '
            coeff_str += str(('Chroma', self.chroma_coeff))
        if 'poly' in self.features:
            if len(coeff_str) != 0:
                coeff_str += ', '
            coeff_str += str(('Poly', self.poly_order))
        if len(coeff_str) != 0:
            f_cfg_str += '\n\tCoeff -> (' + coeff_str + ')'
        return f_cfg_str

    def __repr__(self):
        return str(self)


class ReductionConfig:
    ''''''
    def __init__(self, feature_reduction=None, reduction_size=10,
                 vector_reduction='mean', iqr_coeff=50):
        self.feature_reduction = feature_reduction
        self.reduction_size = reduction_size
        self.vector_reduction = vector_reduction
        self.iqr_coeff = iqr_coeff

    def __str__(self):
        r_cfg_str = 'Reduction Config ->'
        r_cfg_str += '\n\tFeature Reduction -> ' + str(self.feature_reduction)
        if self.feature_reduction != 'lda':
            r_cfg_str += '\n\tReduction size -> ' + str(self.reduction_size)
        r_cfg_str += '\n\tVector Reduction -> ' + self.vector_reduction
        if self.vector_reduction == 'iqr':
            r_cfg_str += '\n\t\t IQR Coeff -> ' + str(self.iqr_coeff)
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
    def __init__(self, files=None,
                 features_cfg=FeaturesConfig(
                     features=['mfcc'], mfcc_coeff=20, chroma_coeff=12, poly_order=1),
                 reduction_cfg=ReductionConfig(
                     feature_reduction='', reduction_size=10, vector_reduction='mean')):
        self.functions = {
            'c_stft': [dict(y=None, sr=None, n_chroma=None),
                       feature.chroma_stft, features_cfg.chroma_coeff],
            'c_cqt': [dict(y=None, sr=None, n_chroma=None),
                      feature.chroma_cqt, features_cfg.chroma_coeff],
            'c_cens': [dict(y=None, sr=None, n_chroma=None),
                       feature.chroma_cens, features_cfg.chroma_coeff],
            'mel_spec': [dict(y=None, sr=None), feature.melspectrogram, 128],
            'mfcc': [dict(y=None, sr=None, n_mfcc=None), feature.mfcc, features_cfg.mfcc_coeff],
            'rmse': [dict(y=None), feature.rmse, 1],
            'bandwidth': [dict(y=None, sr=None), feature.spectral_bandwidth, 1],
            'centroid': [dict(y=None, sr=None), feature.spectral_centroid, 1],
            'contrast': [dict(y=None, sr=None), feature.spectral_contrast, 1],
            'rolloff': [dict(y=None, sr=None), feature.spectral_rolloff, 1],
            'poly': [dict(y=None, sr=None, order=None),
                     feature.poly_features, features_cfg.poly_order+1],
            'tonnetz': [dict(y=None, sr=None), feature.tonnetz, 6],
            'zcr': [dict(y=None), feature.zero_crossing_rate, 1],
            'mean': [[None, 0], np.mean],
            'iqr': [[None, None, 0], np.percentile],
            'var': [[None, 0], np.var],
            'pca': [dict(X=None), PCA],
            'ipca': [dict(X=None), IPCA],
            'kpca': [dict(X=None), KPCA],
            'fica': [dict(X=None), FICA],
            'tsvd': [dict(X=None), TSVD],
            'nmf': [dict(X=None), NMF],
            'spca': [dict(X=None), SPCA],
            'cca': [dict(X=None), CCA],
            'plssvd': [dict(X=None), PLSSVD],
            'lda': [dict(X=None, y=None), LDA]
        }
        if files != None:
            self.data_info = Info(
                files=files, features_config=features_cfg, reduction_config=reduction_cfg, info=[])
            self.classes = len(files)
            self.features_dim = 0
            for ftr in features_cfg.features:
                self.features_dim += self.functions[ftr][2]
                self.data_info.info.append((ftr, self.functions[ftr][2]))
            self.features = np.empty((0, self.features_dim))
            self.labels = np.empty((0, self.classes))
            self.class_labels = np.empty(0, dtype=int)

    def extract(self, class_dirs=None, filename=None):
        ''''''
        if class_dirs != None:
            self.load(filename=filename, class_dirs=class_dirs)
            return copy.deepcopy(self)
        if filename != None:
            self.load(filename=filename)
            return copy.deepcopy(self)
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
                if len(reduction[0]) > 2:
                    reduction[0][1] = self.data_info.reduction_config.iqr_coeff
                vector = reduction[1](*reduction[0])
                self.features = np.vstack([
                    self.features, vector])
                if vector.ndim > 1:
                    self.class_labels = np.append(self.class_labels, np.repeat(i, len(vector[0])))
                    self.labels = np.vstack([
                        self.labels, ind2vec(
                            ind=np.repeat(i, len(vector[0])), classes=self.classes)])
                else:
                    self.class_labels = np.append(self.class_labels, i)
                    self.labels = np.vstack([self.labels, ind2vec(ind=[i], classes=self.classes)])
                done += 1
                print_progress(iteration=done, total=total)
        self.features = np.array(self.features)
        self.labels = np.array(self.labels, dtype='int')
        self.data_info.reduction_config.reduction_size = int(0.8 * self.features_dim)
        if self.data_info.reduction_config.feature_reduction != None:
            params = dict(X=self.features, y=self.class_labels)
            fxn = self.functions[self.data_info.reduction_config.feature_reduction]
            fxn[0] = {k: params[k] for k, v in fxn[0].iteritems()}
            self.features = fxn[1](
                n_components=self.data_info.reduction_config.reduction_size).fit(
                    **fxn[0]).transform(self.features)
            self.features_dim = np.shape(self.features)[1]
        return self

    def save(self, filename=None, class_dirs=None):
        ''''''
        if class_dirs != None:
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
        else:
            np.save(filename + '-features.npy', self.features)
            np.save(filename + '-labels.npy', self.labels)

    def load(self, filename=None, class_dirs=None):
        ''''''
        if class_dirs != None:
            name = filename + '-' + class_dirs[0]
            self.features = np.load(name + '-features.npy')
            self.labels = np.load(name + '-labels.npy')
            for i in range(1, len(class_dirs)):
                name = filename + '-' + class_dirs[i]
                self.features = np.vstack([self.features, np.load(name + '-features.npy')])
                self.labels = np.vstack([self.labels, np.load(name + '-labels.npy')])
        else:
            self.features = np.load(filename + '-features.npy')
            self.labels = np.load(filename + '-labels.npy')
        self.features_dim = np.shape(self.features[0])[0]
        self.classes = np.shape(self.labels[0])[0]

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
