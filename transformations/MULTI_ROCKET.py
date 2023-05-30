import numpy as np
import multirocket
import sys
sys.path.insert(1, '/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/utils/')
sys.path.insert(1, '/home/huseyn/Desktop/roc-inc-hcf/ROCKET-Inception-HCF-main/classifiers/')
from SOFTMAX import Softmax
from utils import load_data
import time



class MultiRocket:

    def __init__(self, xtrain, num_features=40000, pooling='ppv+lspv+mpv+mipv'):
        self.xtrain = xtrain
        self.base_parameters = None
        self.diff1_parameters = None
        self.pooling = pooling
        self.n_features_per_kernel = 4
        self.num_features = num_features / 2  # 1 per transformation
        self.num_kernels = int(self.num_features / self.n_features_per_kernel)
        print(self.num_kernels)

    def get_kernels(self):
        xx = np.diff(self.xtrain, 1)
        return multirocket.fit(self.xtrain, num_features=self.num_kernels, max_dilations_per_kernel=32),\
               multirocket.fit(xx, num_features=self.num_kernels, max_dilations_per_kernel=32)
    
    def transform(self, X, kernels):
        xx = np.diff(X, 1)
        x_train_transform = multirocket.transform(
            X, xx,
            kernels[0], kernels[1],
            self.n_features_per_kernel
        )

        x_train_transform = np.nan_to_num(x_train_transform)
        
        a = int(x_train_transform.shape[1] / 8)
        
        pools = []
        
        for pool in self.pooling.split('+'):
            if pool=='ppv':
                p = np.concatenate((x_train_transform[:,:a] ,x_train_transform[:,a*4:a*5]), axis=1)
                
            elif pool=='lspv':
                p = np.concatenate((x_train_transform[:,a:a*2] ,x_train_transform[:,a*5:a*6]), axis=1)
                
            elif pool=='mpv':
                p = np.concatenate((x_train_transform[:,a*2:a*3] ,x_train_transform[:,a*6:a*7]), axis=1)
                
            elif pool=='mipv':
                p = np.concatenate((x_train_transform[:,a*3:a*4] ,x_train_transform[:,a*7:]), axis=1)
            
            pools.append(p)
        
        return np.concatenate(pools, axis=1)


xtrain, ytrain, xtest, ytest = load_data('Haptics')

start = time.time()
mr = MultiRocket(xtrain, 40000, 'ppv+lspv+mpv+mipv')
kernels = mr.get_kernels()

X = mr.transform(xtrain, kernels)

output_layer = keras.layers.Dense(nb_classes, activation='softmax')(X)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

file_path = self.output_directory + 'best_model.hdf5'

model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

self.callbacks = [reduce_lr, model_checkpoint]

return model

print(time.time() - start)