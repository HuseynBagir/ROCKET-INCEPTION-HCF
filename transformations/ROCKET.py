import numpy as np
import sys
sys.path.insert(1, './home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/utils/')
sys.path.insert(1, './home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/transformations/')
from utils import load_data
from rocket_functions.rocket_functions import rock

# @jitclass(spec)
class ROCKET:

    def __init__(self, length_TS, n_filters, pooling):

        self.length_TS = length_TS
        self.n_filters = n_filters
        self.pooling = pooling

    def get_kernels(self):
        return rock.generate_kernels(input_length=self.length_TS, num_kernels=self.n_filters)
    
    def transform(self, X, kernels):
        return rock.apply_kernels(X=X, kernels=kernels)
    
'''
xtrain,ytrain,xtest,ytest = load_data('Coffee')


leng = xtrain.shape[1]

roc = ROCKET(leng, 100)

kernel = roc.get_kernels()

x = roc.transform(np.array(xtrain),kernel)
'''

