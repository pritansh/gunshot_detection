import gunshot_detection as gd
import numpy as np

dataset = gd.Dataset('./dataset')

files = np.empty(2, dtype='object')
files[0] = dataset.train_files[0][0:3]
files[1] = dataset.train_files[1][0:3]

fe = gd.Features(files)
print(fe)
print(fe.features)
print(fe.labels)