import ml
import numpy as np
import matplotlib.pyplot as plt

dataset = ml.Dataset('./dataset')

tr_mean_f = ml.AudioFeatures(files=dataset.train_files, vector_reduction='mean')
te_mean_f = ml.AudioFeatures(files=dataset.test_files, vector_reduction='mean')
tr_mean_pca_f = ml.AudioFeatures(files=dataset.train_files, feature_reduction='pca',
                                 reduction_size=10, vector_reduction='mean')
te_mean_pca_f = ml.AudioFeatures(files=dataset.test_files, feature_reduction='pca',
                                 reduction_size=10, vector_reduction='mean')
tr_mean_f.save('./features/tr_mean_f')
te_mean_f.save('./features/te_mean_f')
tr_mean_pca_f.save('./features/tr_mean_pca_f')
te_mean_pca_f.save('./features/te_mean_pca_f')

tr_var_f = ml.AudioFeatures(files=dataset.train_files, vector_reduction='var')
te_var_f = ml.AudioFeatures(files=dataset.test_files, vector_reduction='var')
tr_var_pca_f = ml.AudioFeatures(files=dataset.train_files, feature_reduction='pca',
                                 reduction_size=10, vector_reduction='var')
te_var_pca_f = ml.AudioFeatures(files=dataset.test_files, feature_reduction='pca',
                                 reduction_size=10, vector_reduction='var')
tr_var_f.save('./features/tr_var_f')
te_var_f.save('./features/te_var_f')
tr_var_pca_f.save('./features/tr_var_pca_f')
te_var_pca_f.save('./features/te_var_pca_f')

tr_iqr_f = ml.AudioFeatures(files=dataset.train_files, vector_reduction='iqr')
te_iqr_f = ml.AudioFeatures(files=dataset.test_files, vector_reduction='iqr')
tr_iqr_pca_f = ml.AudioFeatures(files=dataset.train_files, feature_reduction='pca',
                                 reduction_size=10, vector_reduction='iqr')
te_iqr_pca_f = ml.AudioFeatures(files=dataset.test_files, feature_reduction='pca',
                                 reduction_size=10, vector_reduction='iqr')
tr_iqr_f.save('./features/tr_iqr_f')
te_iqr_f.save('./features/te_iqr_f')
tr_iqr_pca_f.save('./features/tr_iqr_pca_f')
te_iqr_pca_f.save('./features/te_iqr_pca_f')
