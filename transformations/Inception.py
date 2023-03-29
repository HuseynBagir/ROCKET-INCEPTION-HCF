import keras
import sys
sys.path.insert(1, '/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/utils/')
from utils import load_data
import numpy as np
import tensorflow as tf


class Inception:
    
    def __init__(self, length_TS, pretrained_model, pooling):
        
        self.length_TS = length_TS
        
        self.pretrained_model = keras.models.load_model('/home/huseyn/internship/code/inception_pretrained/' + 
                                                       pretrained_model + '/best_model.hdf5')
        
        self.pooling = pooling
    
    def get_kernels(self):
    
        weights_40 = self.pretrained_model.layers[3].get_weights()
        weights_20 = self.pretrained_model.layers[4].get_weights()
        weights_10 = self.pretrained_model.layers[5].get_weights()
        weights_1  = self.pretrained_model.layers[6].get_weights()
        weights_batch = self.pretrained_model.layers[8].get_weights()
        
        
        input_layer = keras.layers.Input(shape=(self.length_TS, 1))
        
        max_pool = keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(input_layer)
        
        conv1 = keras.layers.Conv1D(filters=32, kernel_size=40, strides=1, padding='same', 
                                    activation=None, use_bias=False)(input_layer)
        
        conv2 = keras.layers.Conv1D(filters=32, kernel_size=20, strides=1, padding='same', 
                                    activation=None, use_bias=False)(input_layer)
        
        conv3 = keras.layers.Conv1D(filters=32, kernel_size=10, strides=1, padding='same', 
                                    activation=None, use_bias=False)(input_layer)
        
        conv4 = keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, padding='same',
                                    activation=None, use_bias=False)(max_pool)
        
        concat = keras.layers.Concatenate(axis=2)([conv1, conv2, conv3, conv4])
        out = keras.layers.BatchNormalization()(concat)
        
        #if self.pooling == 'GAP':
         #   out = keras.layers.GlobalAveragePooling1D()(out)
        
        model = tf.keras.models.Model(inputs=input_layer, outputs=out)
        
        
        model.layers[2].set_weights(weights_40)
        model.layers[3].set_weights(weights_20)
        model.layers[4].set_weights(weights_10)
        model.layers[5].set_weights(weights_1)
        model.layers[7].set_weights(weights_batch)     
        
        return model

    
    def transform(self, X, kernels):
        
        y_pred = kernels.predict(X)
        
        if self.pooling == 'ppv+max':
            pvps = np.mean(y_pred > 0, axis=1) 
            maxs = np.max(y_pred, axis=1)
            
            X_array = np.concatenate([pvps,maxs], axis=1)
            
        elif self.pooling == 'GAP':
            X_array = keras.layers.GlobalAveragePooling1D()(y_pred)
            X_array = X_array.numpy()
            
        elif self.pooling == 'max':
            pass
        
        return X_array

'''
xtrain, ytrain, xtest, ytest = load_data('Coffee')
length_TS = int(xtrain.shape[1])

inc = Inception(length_TS, 'Coffee', 'GAP')

model = inc.get_kernels()

X = inc.transform(xtrain, model)
'''