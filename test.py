import ml
import numpy as np

dataset = ml.Dataset('./dataset')
print dataset

fcd = dict(features=['mfcc'], mfcc_coeff=13)
fc = ml.FeaturesConfig(**fcd)
print fc

rcd = dict(feature_reduction='lda')
rc = ml.ReductionConfig(**rcd)
print rc

train_fe = ml.AudioFeatures(files=dataset.train_files, features_cfg=fc, reduction_cfg=rc).extract()
#train_fe.save('./genres/train')
print train_fe
test_fe = ml.AudioFeatures(files=dataset.test_files, features_cfg=fc, reduction_cfg=rc).extract()
#test_fe.save('./genres/test')
print test_fe

net = ml.MLP(train_fe.features_dim, train_fe.classes, [280, 300, 400, 600, 800])
#net = ml.MLP(train_fe.features_dim, train_fe.classes, [280, 300, 560, 720])
#net = gd.Network(20, 10, [200, 400, 800, 1600, 3200])
print net
net.train(train=train_fe, test=test_fe, epochs=500)
