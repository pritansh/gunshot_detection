import ml

d = ml.Dataset('./dataset')
print d

fcd = dict(features=['mfcc', 'mel_spec'], coeffs=dict(mfcc_coeff=20))
fc = ml.FeaturesConfig(**fcd)
print fc

rc = ml.ReductionConfig()
print rc

train_fe = ml.AudioFeatures(files=d.train_files, features_cfg=fc, reduction_cfg=rc).extract()
test_fe = ml.AudioFeatures(files=d.test_files, features_cfg=fc, reduction_cfg=rc).extract()

ncd = dict(features_dim=train_fe.features_dim, classes=train_fe.classes,
           hidden_units=[280, 300, 560, 720], learn_rate=0.01)
nc = ml.NetworkConfig(**ncd)

net = ml.MLP(nc)
print net

net.train(train=train_fe, test=test_fe, epochs=500)

