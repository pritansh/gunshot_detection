import ml
import numpy as np

dataset = ml.Dataset('./genres/wav')
print dataset

train_fe = ml.AudioFeatures(filename='./genres/train', vector_reduction='mean')
#train_fe.save('./genres/train')
print train_fe
test_fe = ml.AudioFeatures(filename='./genres/test', vector_reduction='mean')
#test_fe.save('./genres/test')
print test_fe

#net = ml.MLP(train_fe.features_dim, train_fe.classes, [280, 300, 400, 600, 800])
net = ml.MLP(train_fe.features_dim, train_fe.classes, [280, 300])
#net = gd.Network(20, 10, [200, 400, 800, 1600, 3200])
print net
net.train(train=train_fe, test=test_fe, epochs=500)
