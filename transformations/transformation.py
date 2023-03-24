import numpy as np

from transformations.ROCKET import ROCKET
from transformations.HCF import HCF
from transformations.Inception import Inception


class Transformation:

    def __init__(self, transformations, length_TS, n_filters_rocket=10000, n_filters_hcf=6, pretrained_model='Coffee'):

        self.transformations = transformations
        self.length_TS = length_TS

        self.n_filters_rocket = n_filters_rocket
        self.n_filters_hcf = n_filters_hcf
        self.pretrained_model = pretrained_model

    def transform(self, xtrain, xtest):

        transformed_xtrain = []
        transformed_xtest = []

        for transformation in self.transformations:

            if transformation == 'ROCKET':
                _transformation = ROCKET(length_TS=self.length_TS, n_filters=self.n_filters_rocket)

            elif transformation == 'HCF':
                _transformation = HCF(length_TS=self.length_TS, n_filters=self.n_filters_hcf)

            #elif transformation == 'Inception':
            #    _transformation = Inception(length_TS=self.length_TS, pretrained_model=self.pretrained_model)
            

            kernels = _transformation.get_kernels()

            transformed_xtrain.append(_transformation.transform(X=xtrain, kernels=kernels))
            transformed_xtest.append(_transformation.transform(X=xtest, kernels=kernels))
        
        return np.concatenate(transformed_xtrain, axis=1), np.concatenate(transformed_xtest, axis=1)