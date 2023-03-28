import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def create_directory(directory_path):
    
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)

def load_data(file_name):
    
    folder_path = '/home/huseyn/internship/UCRArchive_2018/'
    folder_path += (file_name + "/")

    train_path = folder_path + file_name + "_TRAIN.tsv"
    test_path = folder_path + file_name + "_TEST.tsv"

    if (os.path.exists(test_path) <= 0):
        print("File not found")
        return None, None, None, None

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    ytrain = train[:, 0]
    ytest = test[:, 0]

    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)

    return xtrain, ytrain, xtest, ytest

def znormalisation(x):

    stds = np.std(x,axis=1,keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))

def encode_labels(y):
    
    labenc = LabelEncoder()
    
    return labenc.fit_transform(y)

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder


def creat_dir(dir_path):
    
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def datasets_path(path):

    return os.listdir(path)


def load_data(file_name):
    
    
    folder_path = "/home/huseyn/internship/UCRArchive_2018/"
    folder_path += (file_name + '/')
    
    train_path = folder_path + file_name + '_TRAIN.tsv'
    test_path = folder_path + file_name + '_TEST.tsv'
    
    if (os.path.exists(test_path) <= 0):
        print('File not found.')
        return None, None, None, None
    
    train = pd.read_csv(train_path, sep='\t', header=None)
    test = pd.read_csv(test_path, sep='\t', header=None)
 
    ytrain = train.iloc[:, 0:1]
    ytest = test.iloc[:, 0:1]
    
    xtrain = train.iloc[:,1:]
    xtest = test.iloc[:,1:]
    
    return xtrain, ytrain, xtest, ytest


def znormalization(x):

    stds = np.std(x,axis=1)
    
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1)) / stds
    
    return (x - x.mean(axis=1)) / x.std(axis=1)


def visualize(x, y, row=0):
    
    return print('class:', y[0][row], plt.plot(x.iloc[row,:]), plt.title('Chinatown Class: ' + str(y[0][row])))


def label_encoder(y):
    
    le = LabelEncoder()
    
    return le.fit_transform(y)
'''



'''
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def create_directory(directory_path):
    
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)

def load_data(file_name):
    
    folder_path = "/mnt/nfs/ceres/bla/archives/new/UCRArchive_2018/UCRArchive_2018/"
    folder_path += (file_name + "/")

    train_path = folder_path + file_name + "_TRAIN.tsv"
    test_path = folder_path + file_name + "_TEST.tsv"

    if (os.path.exists(test_path) <= 0):
        print("File not found")
        return None, None, None, None

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    ytrain = train[:, 0]
    ytest = test[:, 0]

    xtrain = np.delete(train, 0, axis=1)
    xtest = np.delete(test, 0, axis=1)

    return xtrain, ytrain, xtest, ytest

def znormalisation(x):

    stds = np.std(x,axis=1,keepdims=True)
    if len(stds[stds == 0.0]) > 0:
        stds[stds == 0.0] = 1.0
        return (x - x.mean(axis=1, keepdims=True)) / stds
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True))

def encode_labels(y):
    
    labenc = LabelEncoder()
    
    return labenc.fit_transform(y)
'''