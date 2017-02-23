import gunshot_detection as gd
import numpy as np

dataset = gd.Dataset('./dataset')
print dataset

train_fe = gd.Features(dataset.train_files, vector_reduction='mean')
print train_fe
test_fe = gd.Features(dataset.test_files, vector_reduction='mean')
print test_fe

net = gd.Network(train_fe.features_dim, train_fe.classes, [280, 300, 400, 600, 800])
#net = gd.Network(20, 10, [200, 400, 800, 1600, 3200])
print net
net.train(train=train_fe, test=test_fe, epochs=500)