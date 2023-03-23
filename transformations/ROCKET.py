import numpy as np

from transformations import rocket_functions

# @jitclass(spec)
class ROCKET:

    def __init__(self, length_TS, n_filters):

        self.length_TS = length_TS
        self.n_filters = n_filters

    def get_kernels(self):
        return rocket_functions.generate_kernels(input_length=self.length_TS, num_kernels=self.n_filters)
    
    def transform(self, X, kernels):
        return rocket_functions.apply_kernels(X=X, kernels=kernels)