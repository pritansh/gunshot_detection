import random
import os
import numpy as np
from librosa import load, feature

def get_dataset(dataset_dir='./dataset', train_perc=0.8, valid_perc=0.1, test_perc=0.1):
    '''
    Get all files from the dataset directory and after randomizing it
    splits it into training, validation and testing parts based on
    train_perc -> training part % (Default - 0.8[80%])
    valid_perc -> validation part % (Default - 0.1[10%])
    test_perc -> testing part % (Default - 0.1[10%])
    '''
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
    '''
    Extracts features from the audio file
    mfcc - mel frequency cepstral coefficients (20 coefficients)
    '''
    samples, rate = np.array(load(audio_path))
    mfcc = np.mean(feature.mfcc(y=samples, sr=rate).T, axis=0)
    return mfcc

def get_data(files, feature_size=20):
    '''
    Extracts features for all data and arranges features and labels
    '''
    length = [len(e) for e in files]
    total = sum(length)
    features = np.empty((total, feature_size))
    labels = np.zeros(total)
    itr = 0
    for i in range(0, len(length)):
        for j in range(0, length[i]):
            features[itr+j, :] = extract_features(files[i][j])
        labels[itr:itr+length[i]] = i+1
        itr += length[i]
    return features, labels

def get_dataset_features(train_files, valid_files, test_files):
    '''
    Returns all training, validation and testing features and labels
    '''
    print 'Extracting training data'
    train = get_data(train_files)
    print 'Extracting validation data'
    valid = get_data(valid_files)
    print 'Extracting testing data'
    test = get_data(test_files)
    return train, valid, test

dataset_files = get_dataset()
data = get_dataset_features(*dataset_files)
