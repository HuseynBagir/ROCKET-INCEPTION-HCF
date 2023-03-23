import keras
from utils import load_data
import numpy as np
import pandas as pd
import tensorflow as tf


class InceptionModule:
    
    def __init__(self, length_TS, pretrained_model):
        
        xtrain, ytrain, xtest, ytest = load_data(length_TS)
        
        self.length_TS = xtrain.shape[1]
        
        self.pretrained_model = keras.models.load_model('/home/huseyn/internship/code/inception_pretrained/' + 
                                                       pretrained_model + '/best_model.hdf5')
    

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
        batch = keras.layers.BatchNormalization()(concat)
        
        model = tf.keras.models.Model(inputs=input_layer, outputs=batch)
        
        
        model.layers[2].set_weights(weights_40)
        model.layers[3].set_weights(weights_20)
        model.layers[4].set_weights(weights_10)
        model.layers[5].set_weights(weights_1)
        model.layers[7].set_weights(weights_batch)     
        
        return model

    
    def transform(self, X, kernels):
        
        xtrain, ytrain, xtest, ytest = load_data(X)
        y_pred = kernels.predict(xtrain)
        pvps = np.mean(y_pred > 0, axis=(0, 1)) 
        maxs = np.max(y_pred, axis=(0, 1)) 
        
        return pvps, maxs








