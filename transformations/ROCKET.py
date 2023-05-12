import numpy as np
import rocket_functions
import sys
sys.path.insert(1, '/home/hbagirov/inetrnship/projects/ROCKET-Inception-HCF-main/utils/')
from utils import load_data
import time

class ROCKET:

    def __init__(self, length_TS, n_filters, pooling='ppv+max+GAP'):

        self.length_TS = length_TS
        self.n_filters = n_filters
        self.pooling = pooling

    def get_kernels(self):
        return rocket_functions.generate_kernels(input_length=self.length_TS, num_kernels=self.n_filters)
    
    def transform(self, X, kernels):
        
        X_arr =  rocket_functions.apply_kernels(X=X, kernels=kernels)
        
        pools=[]
        
        for pool in self.pooling.split('+'):
            if pool == 'ppv':
                p = X_arr[:,::6]
                
            elif pool == 'max':
                p = X_arr[:,1::6]
                
            elif pool == 'GAP':
                p = X_arr[:,2::6]
                
            elif pool == 'mpv':
                p = X_arr[:,3::6]
                
            elif pool == 'mipv':
                p = X_arr[:,4::6]
                
            elif pool == 'lspv':
                p = X_arr[:,5::6]
                
            pools.append(p)
            
        return np.concatenate(pools, axis=1)
        
'''xtrain, ytrain, xtest, ytest = load_data('Coffee')
length_TS = int(xtrain.shape[1])
hcf = ROCKET(length_TS, 500, 'ppv+max+GAP+mpv+lspv+mipv')
model = hcf.get_kernels()

start = time.time()
X = hcf.transform(xtrain, model)
print(time.time() - start)'''



