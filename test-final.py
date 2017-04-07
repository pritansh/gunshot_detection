import ml

dataset = ml.Dataset('./dataset')

sets = [['c_stft', 'c_cqt', 'c_cens'], ['mel_spec', 'mfcc'], ['bandwidth', 'centroid', 'rolloff'], ['rmse', 'poly', 'tonnetz', 'zcr']]

coeffs = dict(chroma_coeff=12, mfcc_coeff=20, poly_order=1)

rcd_1 = dict(vector_reduction='mean', feature_reduction=None)
rc_1 = ml.ReductionConfig(**rcd_1)

rcd_2 = dict(vector_reduction='mean', feature_reduction='pca', reduction_size=16)
rc_2 = ml.ReductionConfig(**rcd_2)

rcd_3 = dict(vector_reduction='mean', feature_reduction='lda')
rc_3 = ml.ReductionConfig(**rcd_3)

fcd_1 = dict(features=sets[0], chroma_coeff=coeffs['chroma_coeff'])
fc_1 = ml.FeaturesConfig(**fcd_1)

fcd_2 = dict(features=sets[1], mfcc_coeff=coeffs['mfcc_coeff'])
fc_2 = ml.FeaturesConfig(**fcd_2)

fcd_3 = dict(features=sets[2])
fc_3 = ml.FeaturesConfig(**fcd_3)

fcd_4 = dict(features=sets[3], poly_order=coeffs['poly_order'])
fc_4 = ml.FeaturesConfig(**fcd_4)

fc = [fc_1, fc_2, fc_3, fc_4]
rc = [rc_1, rc_2, rc_3]

accuracy = []

for i in range(0, len(fc)):
    for j in range(0, len(rc)):
        print fc[i], rc[j]
        train = ml.AudioFeatures(files=dataset.train_files, features_cfg=fc[i], reduction_cfg=rc[j]).extract()
        test = ml.AudioFeatures(files=dataset.test_files, features_cfg=fc[i], reduction_cfg=rc[j]).extract()
        ncd = dict(features_dim=train.features_dim, classes=train.classes, hidden_units=[280, 300, 560, 720], learn_rate=0.01)
        nc = ml.NetworkConfig(**ncd)
        net = ml.MLP(nc)
        print net
        accuracy.append(net.train(train=train, test=test, epochs=500))

print accuracy
        