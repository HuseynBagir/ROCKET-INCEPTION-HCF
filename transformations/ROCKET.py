import numpy as np
import sys
sys.path.insert(1, './home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/utils/')
sys.path.insert(1, './home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/transformations/')
from utils import load_data
import rocket_functions

# @jitclass(spec)
class ROCKET:

    def __init__(self, length_TS, n_filters):

        self.length_TS = length_TS
        self.n_filters = n_filters

    def get_kernels(self):
        return rocket_functions.generate_kernels(input_length=self.length_TS, num_kernels=self.n_filters)
    
    def transform(self, X, kernels):
        return rocket_functions.apply_kernels(X=X, kernels=kernels)
    
    
xtrain,ytrain,xtest,ytest = load_data('Coffee')


leng = xtrain.shape[1]

roc = ROCKET(leng, 100)

kernel = roc.get_kernels()

x = roc.transform(np.array(xtrain),kernel)


