import keras
import numpy as np
import sys
sys.path.insert(1, '/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/utils/')
from utils import load_data

class Inception:
    
    def __init__(self, length_TS, pretrained_model, pooling):
        
        self.length_TS = length_TS
        
        self.pretrained_model = keras.models.load_model('/home/huseyn/internship/code/inception_pretrained/' + 
                                                       pretrained_model + '/best_model.hdf5')
        
        self.pooling = pooling
    
    def get_kernels(self):
    
        layer_names = []

        for layer in self.pretrained_model.layers:
            if 'input_1' in layer.name:
                continue
            if 'global_average_pooling1d' in layer.name:
                continue
            if 'dense' in layer.name:
                continue
            
            #print(layer.name)
            layer_names.append(layer.name)
        
        selected_layers = [self.pretrained_model.get_layer(name) for name in layer_names]
            
        model = keras.models.Model(inputs=selected_layers[0].input, outputs=selected_layers[-1].output)
        
        return model


    def transform(self, X, kernels):
        
        y_pred = kernels.predict(X)
        
        pools = []
        
        for pool in self.pooling.split('+'):
            
            if pool == 'max':
                p = np.max(y_pred, axis=1)
                
            elif pool == 'ppv':
                p = np.mean(y_pred > 0, axis=1) 
        
            elif pool == 'GAP':
                p = np.mean(y_pred, axis=1)
        
            pools.append(p)
            
        X_array = np.concatenate(pools, axis=1)
        
        return X_array