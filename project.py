import random, os, numpy as np
from librosa import load, feature

def get_dataset(dataset_dir='./dataset', train_perc=0.8, valid_perc=0.1, test_perc=0.1):
    train_files = []
    valid_files = []
    test_files = []
    for root, dirs, files in os.walk(dataset_dir):
        data_files = [os.path.join(root, filename) for filename in files]
        random.shuffle(data_files)
        size = len(data_files)
        train_size = int(size * train_perc)
        valid_size = int(size * valid_perc)
        train_files.append(data_files[:valid_size+train_size])
        valid_files.append(data_files[:valid_size])
        test_files.append(data_files[valid_size+train_size:])
    return train_files[1:], valid_files[1:], test_files[1:]

def extract_features(audio_path):
    samples, rate = np.array(load(audio_path))
    mfcc = np.mean(feature.mfcc(y=samples, sr=rate).T, axis=0)
    return mfcc

def get_dataset_features(train_files, valid_files, test_files):
    train_features = np.zeros(20)
    valid_features = np.zeros(20)
    test_features = np.zeros(20)
    for classes in range(0, len(train_files)):
        for file_path in train_files[classes]:
            train_features = np.vstack((train_features, extract_features(file_path)))
            print np.shape(train_features)
        for file_path in valid_features[classes]:
            valid_features = np.vstack((valid_features, extract_features(file_path)))
            print np.shape(valid_features)
        for file_path in test_features[classes]:
            test_features = np.vstack((test_features, extract_features(file_path)))
            print np.shape(test_features)
    return train_features, valid_features, test_features

train_files, valid_files, test_files = get_dataset()
print np.shape(train_files), np.shape(valid_files), np.shape(test_files)
train, valid, test = get_dataset_features(train_files, valid_files, test_files)
print np.shape(train), np.shape(valid), np.shape(test)