import gunshot_detection as gd
import numpy as np

dataset = gd.Dataset('./dataset')
print dataset

train_fe = gd.Features(dataset.train_files)
print train_fe
test_fe = gd.Features(dataset.test_files)
print test_fe

net = gd.Network(20, 2, [280, 300, 400, 600, 800])
print net
net.train(train=train_fe, test=test_fe, epochs=500)