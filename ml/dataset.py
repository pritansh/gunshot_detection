import os
import numpy as np

class Dataset:
    ''''''
    def __init__(self, directory_path='', dir_comb=[], train_perc=0.60):
        self.class_dirs = os.listdir(directory_path)
        if len(dir_comb) > 0:
            self.class_dirs = dir_comb
        classes = len(self.class_dirs)
        self.train_files = np.empty(classes, dtype='object')
        self.test_files = np.empty(classes, dtype='object')
        for i in range(0, classes):
            dir_path = directory_path + '/' + self.class_dirs[i]
            files = np.core.defchararray.add(dir_path + '/', os.listdir(dir_path))
            np.random.shuffle(files)
            train_size = int(len(files) * train_perc)
            self.train_files[i] = files[:train_size]
            self.test_files[i] = files[train_size:]

    def __str__(self):
        data_str = 'Dataset -> '
        for i in range(0, len(self.class_dirs)):
            data_str += '\n\t' + str(i+1) + '->' + self.class_dirs[i]
            data_str += ' ' + str((len(self.train_files[i]), len(self.test_files[i])))
        return data_str

    def __repr__(self):
        return str(self)
