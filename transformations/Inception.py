import keras
import sys
sys.path.insert(1, '/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/utils/')
from utils import load_data
import numpy as np
import tensorflow as tf
from itertools import permutations


class Inception:
    
    def __init__(self, length_TS, pretrained_model, pooling):
        
        self.length_TS = length_TS
        
        self.pretrained_model = keras.models.load_model('/home/huseyn/internship/code/inception_pretrained/' + 
                                                       pretrained_model + '/best_model.hdf5')
        
        self.pooling = pooling
    
    def get_kernels(self):
    
        layer_names = ['input_1', 'reshape', 'max_pooling1d', 'conv1d', 'conv1d_1', 'conv1d_2', 'conv1d_3', 'concatenate', 
                       'batch_normalization', 'activation','conv1d_4', 'max_pooling1d_1', 'conv1d_5', 'conv1d_6', 'conv1d_7', 
                       'conv1d_8', 'concatenate_1', 'batch_normalization_1','activation_1', 'conv1d_9', 'max_pooling1d_2', 
                       'conv1d_10', 'conv1d_11', 'conv1d_12', 'conv1d_13', 'concatenate_2', 'conv1d_14',
                       'batch_normalization_2', 'batch_normalization_3', 'activation_2', 'add', 'activation_3', 'conv1d_15', 
                       'max_pooling1d_3', 'conv1d_16','conv1d_17', 'conv1d_18', 'conv1d_19', 'concatenate_3', 
                       'batch_normalization_4', 'activation_4', 'conv1d_20','max_pooling1d_4', 'conv1d_21', 'conv1d_22', 
                       'conv1d_23', 'conv1d_24', 'concatenate_4', 'batch_normalization_5', 'activation_5',
                       'conv1d_25', 'max_pooling1d_5', 'conv1d_26', 'conv1d_27', 'conv1d_28', 'conv1d_29', 'concatenate_5', 
                       'conv1d_30','batch_normalization_6', 'batch_normalization_7', 'activation_6', 'add_1']
        
        selected_layers = [self.pretrained_model.get_layer(name) for name in layer_names]
            
        model = keras.models.Model(inputs=selected_layers[0].input, outputs=selected_layers[-1].output)
        
        return model

    
    def transform(self, X, kernels):
        
        y_pred = kernels.predict(X)
        
        if tuple(self.pooling.split('+')) in list(permutations(['ppv'])):
            pvps = np.mean(y_pred > 0, axis=1)
            
            X_array = pvps
            
        elif tuple(self.pooling.split('+')) in list(permutations(['max'])):
            maxs = np.max(y_pred, axis=1)
            
            X_array = maxs
            
        elif tuple(self.pooling.split('+')) in list(permutations(['GAP'])):
            GAP = np.mean(y_pred, axis=1) 
            
            X_array = GAP
            
        elif tuple(self.pooling.split('+')) in list(permutations(['GAP', 'max'])):
            GAP = np.mean(y_pred, axis=1) 
            maxs = np.max(y_pred, axis=1)
            
            X_array = np.concatenate([GAP,maxs], axis=1)

        elif tuple(self.pooling.split('+')) in list(permutations(['ppv', 'max'])):
            pvps = np.mean(y_pred > 0, axis=1) 
            maxs = np.max(y_pred, axis=1)
            
            X_array = np.concatenate([pvps,maxs], axis=1)
            
        elif tuple(self.pooling.split('+')) in list(permutations(['ppv', 'GAP'])):
            pvps = np.mean(y_pred > 0, axis=1) 
            GAP = np.mean(y_pred, axis=1)
            
            X_array = np.concatenate([pvps,GAP], axis=1)
            
        elif tuple(self.pooling.split('+')) in list(permutations(['ppv', 'max', 'GAP'])):
            pvps = np.mean(y_pred > 0, axis=1) 
            maxs = np.max(y_pred, axis=1)
            GAP = np.mean(y_pred, axis=1)
            
            X_array = np.concatenate([pvps,maxs,GAP], axis=1)
        
        return X_array

'''
xtrain, ytrain, xtest, ytest = load_data('Coffee')
length_TS = int(xtrain.shape[1])

inc = Inception(length_TS, 'Coffee', 'ppv')

model = inc.get_kernels()

X = inc.transform(xtrain, model)
'''