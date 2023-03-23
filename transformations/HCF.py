import tensorflow as tf
import numpy as np

class HCF:

    def __init__(self, length_TS, n_filters=6):

        self.length_TS = length_TS
        
        self.n_filters = n_filters

        self.increasing_trend_kernels = [2**i for i in range(1,self.n_filters + 1)]
        self.decreasing_trend_kernels = [2**i for i in range(1,self.n_filters + 1)]
        self.peak_kernels = [2**i for i in range(2,self.n_filters + 1)]

        print(self.increasing_trend_kernels)
        print(self.decreasing_trend_kernels)
        print(self.peak_kernels)

    def increasing_trend_filter(self, kernel_size):
        
        '''
        Function to return a temporaty model that applies the custom increasing detection filter of length f.
        
        Args:
        
            f : length of filter (int), no default value
            input_length : length of input time series to create tensorflow model
        '''

        if kernel_size % 2 > 0:
            raise ValueError("Filter size should be even.")

        input_layer = tf.keras.layers.Input((self.length_TS,1))

        output_layer = tf.keras.layers.Conv1D(filters=1,kernel_size=kernel_size,padding='same',use_bias=False)(input_layer) # do not use bias because custom filters are non biased convolution filters
        output_layer = tf.keras.layers.Activation(activation='relu')(output_layer) # apply ReLU to remove the non activated part (negative part) of the filter response

        model = tf.keras.models.Model(inputs=input_layer,outputs=output_layer) # create the model

        filter_ = np.ones(shape=(kernel_size,1,1)) # define the filter weights with the shape corresponding the Conv1D layer in keras (kernel_size, input_channels, output_channels)
        indices_ = np.arange(kernel_size)

        filter_[indices_ % 2 == 0] *= -1 # formula of increasing detection filter

        model.layers[1].set_weights([filter_]) # set the filter on the model

        return model

    def decreasing_trend_filter(self, kernel_size):
        
        '''
        Function to return a temporaty model that applies the custom decreasing detection filter of length f.
        Args:
        
            f : length of filter (int), no default value
            input_length : length of input time series to create tensorflow model
        '''

        if kernel_size % 2 > 0:
            raise ValueError("Filter size should be even.")

        input_layer = tf.keras.layers.Input((self.length_TS,1))

        output_layer = tf.keras.layers.Conv1D(filters=1,kernel_size=kernel_size,padding='same',use_bias=False)(input_layer) # do not use bias because custom filters are non biased convolution filters
        output_layer = tf.keras.layers.Activation(activation='relu')(output_layer) # apply ReLU to remove the non activated part (negative part) of the filter response

        model = tf.keras.models.Model(inputs=input_layer,outputs=output_layer) # create the model

        filter_ = np.ones(shape=(kernel_size,1,1)) # define the filter weights with the shape corresponding the Conv1D layer in keras (kernel_size, input_channels, output_channels)
        indices_ = np.arange(kernel_size)

        filter_[indices_ % 2 > 0] *= -1 # formula of increasing detection filter

        model.layers[1].set_weights([filter_]) # set the filter on the model

        return model

    def peak_filter(self, kernel_size):

        '''
        Function to return a temporaty model that applies the custom peak detection filter of length f.
        Args:
        
            f : two times the sub filter length (int), no default value, the function takes as input 
                the length of two sub part of the outcom filter. The outcome filter
                has the length of f + f//2 (3 sub filters of length f//2)
            input_length : length of input time series to create tensorflow model
        '''

        if kernel_size % 2 > 0:
            raise ValueError("Filter size should be even.")

        input_layer = tf.keras.layers.Input((self.length_TS,1))

        output_layer = tf.keras.layers.Conv1D(filters=1,kernel_size=kernel_size+kernel_size//2,padding='same',use_bias=False)(input_layer) # do not use bias because custom filters are non biased convolution filters
        output_layer = tf.keras.layers.Activation(activation='relu')(output_layer) # apply ReLU to remove the non activated part (negative part) of the filter response

        model = tf.keras.models.Model(inputs=input_layer,outputs=output_layer) # create the model

        if kernel_size == 2:

            filter_ = np.asarray([-1,2,-1]).reshape((-1,1,1)) # if f == 2 then the filter will be [-1,2,-1]
        
        else:

            filter_ = np.zeros(shape=(kernel_size+kernel_size//2,1,1)) # define the filter weights with the shape corresponding the Conv1D layer in keras (kernel_size, input_channels, output_channels)
            # filter_[0:f//2] = np.linspace(start=0,stop=1,num=f//2+1)[1:].reshape((-1,1,1))
            # filter_[f//2:f] = -1
            # filter_[f:] = np.linspace(start=filter_[f//2-1,0,0],stop=0,num=f//2+1)[:-1].reshape((-1,1,1))

            xmesh = np.linspace(start=0,stop=1,num=kernel_size//4+1)[1:].reshape((-1,1,1)) # define the xmesh used to construct the parabolic shape of the filter, the length of the xmesh is half the length of each sub part of the filter hence f//4

            filter_left = xmesh**2 # project the mesh on a parabolic function
            filter_right = filter_left[::-1] # inverse the filter_left to construct the right one

            # use filter_left and filter_right to constuct the outcome filter

            # the first part is in the negative part

            filter_[0:kernel_size//4] = -filter_left
            filter_[kernel_size//4:kernel_size//2] = -filter_right

            # the second part is in the positive part by with a double amplitude

            filter_[kernel_size//2:3*kernel_size//4] = 2 * filter_left
            filter_[3*kernel_size//4:kernel_size] = 2 * filter_right

            # the third part is in the negative part

            filter_[kernel_size:5*kernel_size//4] = -filter_left
            filter_[5*kernel_size//4:] = -filter_right

        model.layers[1].set_weights([filter_]) # set the filter on the model

        return model
    
    def get_kernels(self):

        kernels = []

        for kernel_size in self.increasing_trend_kernels:
            kernels.append(self.increasing_trend_filter(kernel_size=kernel_size))
        
        for kernel_size in self.decreasing_trend_kernels:
            kernels.append(self.decreasing_trend_filter(kernel_size=kernel_size))
        
        for kernel_size in self.peak_kernels:
            kernels.append(self.peak_filter(kernel_size=kernel_size))
        
        return kernels

    def transform(self, X, kernels):

        n = int(X.shape[0])
        X = np.expand_dims(X, axis=2)

        #get the output number of channels needed
        m = len(self.increasing_trend_kernels) + len(self.decreasing_trend_kernels) + len(self.peak_kernels)

        X_transformed = np.zeros(shape=(n, m*2)) # define the transformed input ndarray

        i = 0

        for kernel in kernels:

            y = np.asarray(kernel(X)).reshape(n,self.length_TS)

            X_transformed[:,i] = np.sum(np.heaviside(y, 0),axis=1) / (self.length_TS * 1.0)
            X_transformed[:,i+1] = np.max(y)
            
            i += 2

        return X_transformed