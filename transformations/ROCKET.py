import numpy as np
import rocket_functions

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
                p = X_arr[:,::3]
                
            elif pool == 'max':
                p = X_arr[:,1::3]
                
            elif pool == 'GAP':
                p = X_arr[:,2::3]
                
            pools.append(p)
            
        return np.concatenate(pools, axis=1)
