import ml

dir_path = './final/kartik/'

train = ml.AudioFeatures().extract(filename=dir_path + 'final-train')
test = ml.AudioFeatures().extract(filename=dir_path + 'final-test')

ncd = dict(features_dim=21, classes=2, hidden_units=[80, 100], learn_rate=0.01)
nc = ml.NetworkConfig(**ncd)

net = ml.MLP(nc)

print net

net.train(train=train, test=test, epochs=5000)