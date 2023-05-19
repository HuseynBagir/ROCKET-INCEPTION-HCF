import numpy as np

import sys
sys.path.insert(1, '/home/hbagirov/inetrnship/projects/ROCKET-Inception-HCF-main/utils/')
sys.path.insert(1, '/home/hbagirov/inetrnship/projects/ROCKET-Inception-HCF-main/transformations/')


from HCF import HCF
from Inception import Inception
from MULTI_ROCKET import  MultiRocket
from ROCKET import ROCKET

from utils import load_data, znormalisation, create_directory, encode_labels


class Transformation:

    def __init__(self, transformations, xtrain, length_TS, n_filters_rocket=10000, num_features=40000,
                 n_filters_hcf=6, pretrained_model='Coffee', pooling='ppv+lspv+mpv+mipv'):

        self.transformations = transformations
        self.xtrain = xtrain
        self.length_TS = length_TS
        self.num_features = num_features
        self.n_filters_rocket = n_filters_rocket
        self.n_filters_hcf = n_filters_hcf
        self.pretrained_model = pretrained_model
        self.pooling = pooling

    def transform(self, xtrain, xtest):

        transformed_xtrain = []
        transformed_xtest = []

        for transformation in self.transformations:

            if transformation == 'ROCKET':
                _transformation = ROCKET(length_TS=self.length_TS, n_filters=self.n_filters_rocket, pooling=self.pooling)
            
            elif transformation == 'MultiROCKET':
                _transformation = MultiRocket(xtrain=self.xtrain, num_features=self.num_features, pooling=self.pooling)
            
            elif transformation == 'HCF':
                _transformation = HCF(length_TS=self.length_TS, n_filters=self.n_filters_hcf, pooling=self.pooling)

            elif transformation == 'Inception':
                _transformation = Inception(length_TS=self.length_TS, pretrained_model=self.pretrained_model, pooling=self.pooling)
            

            kernels = _transformation.get_kernels()

            transformed_xtrain.append(_transformation.transform(X=xtrain, kernels=kernels))
            transformed_xtest.append(_transformation.transform(X=xtest, kernels=kernels))
        
        return np.concatenate(transformed_xtrain, axis=1), np.concatenate(transformed_xtest, axis=1)




