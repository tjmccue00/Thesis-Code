"""GaussianNN

This module allows the user to create a separable Gaussian neural network.
    
    Author: Siyuan (Simon) Xing
    Email: sixing@calpoly.edu
    Licence: MIT Licence
    Copyright (c) 2022
    Version: 1.0.6  
    Version: 1.0.5 (split the trainable configuration for mean and)
"""

import tensorflow as tf
import numpy as np


class GaussianNet(tf.keras.Model):
    """Separable Gaussian neural networks.
        A NN class for separable-Gaussian NNs. The input data is splitted by its dimensions and fed sequentially to each layer.

        Attributes -
            _hidden_layer_num: number of hidden layers. 
            _hidden_layers: an array of hidden_layers.
            _output_layer:  the output layer.
        
        Examples -
            1. Random trainable weights 
                my_model = GaussianNN.GaussianNet(mu_grid, sigma_arr)

                    By default,  random weight initializers are used; all variables (mean, variance, weight) are trainable. 
                
            2.  Init with some untrainable layers
                my_model = GaussianNN.GaussianNet(mu_grid, sigma_arr, 
                                    weight_initializers=layer_weight_initializers,
                                    weights_untrainable_layers=[1], 
                                    center_untrainable_layers=[1], 
                                    width_untrainable_layers=[1])
                where 1 is the layer index (starts from 1).
            
    """
    def __init__(self, mu_arr_list, sigma_arr_list, output_layer_neuron_num, weight_initializers=[], weights_untrainable_layers=[], center_untrainable_layers=[], width_untrainable_layers=[], data_type='float32'):
        """Constructor.
        
        Args:
            mu_arr_list: the list of mean arrays.
            sigma_arr_list: the list of variance arrays.
            weight_initializers: the dictionary of layer initializers. One shall only specify the layers with initializers other than random initializers.
            weights_untrainable_layers: the list of layers whose weights are untrainable. The layer index starts from 1.
            center_untrainable_layers: the list of layers whose mean are untrainable. The layer index starts from 1.
            width_untrainable_layers: the list of layers whose variance are untrainable. The layer index starts from 1. 
        """
        super(GaussianNet, self).__init__()

        if len(mu_arr_list) != len(sigma_arr_list):
            raise Exception('The lenth of mu, sigma arrays have to be identical.')
        if len(mu_arr_list) < 2:
            raise Exception('The variable dimensions should be greater or equal to two.')

        self._hidden_layer_num = len(mu_arr_list)  # layer numbers
        #pre-processing
        hidden_layer_mean_trainibility, hidden_layer_variance_trainibility = self.getHiddenLayerCenterandWidthtrainablility(center_untrainable_layers, width_untrainable_layers)

        weight_trainibility, my_weight_initializers = self.getLayerWeightsTrainabilityAndInitializer(weights_untrainable_layers, weight_initializers)

        #create hidden and output layers
        self._hidden_layers = self.createHiddenLayers(mu_arr_list, sigma_arr_list, weight_trainibility,
                                        hidden_layer_mean_trainibility, hidden_layer_variance_trainibility, my_weight_initializers[:-1], data_type)        

        self._output_layer = tf.keras.layers.Dense(output_layer_neuron_num, kernel_initializer = my_weight_initializers[-1], use_bias=False, trainable=weight_trainibility[-1]) #output linear layer, not trainable, unit weights


    def call(self, inputs):
        splitted_data = tf.split(inputs, self._hidden_layer_num, axis=-1)

        xn = self._hidden_layers[0](splitted_data[0])
        for i in range(1, self._hidden_layer_num):
            xn = self._hidden_layers[i]([xn, splitted_data[i]])
        outputs = self._output_layer(xn) 

        return outputs

    #utilities
    def createHiddenLayers(self, mu_arr_list, sigma_arr_list, weight_trainibility_arr, mean_trainibility_arr, variance_trainibility_arr, weight_initializer_list, dtype):
        layers = []

        layers.append(FirstGaussian(mu_arr_list[0], sigma_arr_list[0],
                                        mean_isTrainable = mean_trainibility_arr[0], variance_isTrainable=variance_trainibility_arr[0], data_type=dtype))

        for i in range(self._hidden_layer_num - 1):
            layers.append(Gaussian(len(mu_arr_list[i]), len(mu_arr_list[i + 1]), mu_arr_list[i + 1],
                sigma_arr_list[i + 1], w_init = weight_initializer_list[i+1], mean_isTrainable = mean_trainibility_arr[i + 1], 
                variance_isTrainable = variance_trainibility_arr[i+1], weight_trainable = weight_trainibility_arr[i + 1], data_type=dtype)) 

        return layers

    def getHiddenLayerCenterandWidthtrainablility(self, center_untrainable_layers, width_untrainable_layers):
        mean_trainibility = [True] * self._hidden_layer_num 
        vaiance_trainibility = [True] * self._hidden_layer_num 

        for (untrain_mean_idx, untrain_variance_idx) in zip(center_untrainable_layers, width_untrainable_layers):
            mean_trainibility[untrain_mean_idx-1] = False
            vaiance_trainibility[untrain_variance_idx-1]=False

        return mean_trainibility, vaiance_trainibility
    
    def getLayerWeightsTrainabilityAndInitializer(self, weights_untrainable_layers,  initializerDic):
        total_layer_num = self._hidden_layer_num + 1
        weight_trainibility = [True] * (total_layer_num)
        weight_trainibility[0] = False # The first layer has no weights
        for idx in weights_untrainable_layers:
            weight_trainibility[idx-1] = False

        #by default, last layer unit init weights, other layers random init weights.
        weights_initializer = [tf.initializers.random_normal()] * (total_layer_num)
        weights_initializer[-1] = tf.initializers.ones()

        #update user input 
        for (key, value) in initializerDic.items():
            weights_initializer[key-1] = value

        return weight_trainibility, weights_initializer


class FirstGaussian(tf.keras.layers.Layer):
    """The first Gaussian-radial-basis layers, which only accept one input.

    Attributes:
        mu_arr: the array of expected values.
        sigma_arr: the array of variance.

    Example:
        1. Trainable centers and widths
            gaussian_layer = FirstGaussian([1,2,3], [1,1,1]) 
        2. untrainable centers and widths
            gaussian_layer = FirstGaussian([1,2,3], [1,1,1], False, False)
        3. data types other than float32
            gaussian_layer = FirstGaussian([1,2,3], [1,1,1], 'float64') 


    """
    def __init__(self, mu, sigma, mean_isTrainable=True, variance_isTrainable=True, data_type='float32'):
        """Constructor.
        The first layer does not have weights. 
        Args:
            mu_arr: the array of expected values.
            sigma_arr: the array of variance.
            mean_isTrainable: True if centers are trainable.
            variance_isTrainable: True if widths are trainable.
        """
        super(FirstGaussian, self).__init__()
        self.data_type = data_type
        self.mu = tf.Variable(initial_value=mu, trainable=mean_isTrainable, dtype=data_type, name = 'mu')
        self.sigma = tf.Variable(initial_value=sigma, trainable=variance_isTrainable, dtype=data_type, name = 'sigma')
      

    def call(self, inputs):
        """Forward-pass action for the first gaussian layer.

        Args:
            inputs: N-dimensional spatial point
        Return:
            output of the Gaussian layer.
        """
        return tf.exp(tf.constant(-0.5, dtype=self.data_type)*(inputs - self.mu)**2/self.sigma**2)
         

class Gaussian(tf.keras.layers.Layer):
    """Customized layers. This is for layers other than the first layer.
     The layer is composed of 1-D Gaussian-radial-basis functions used for a coordinate
     component dimension.

    Attributes:
        mu_arr: the array of mean values.
        sigma_arr: the array of variance.
        w: weights.

    Example:
    """
    def __init__(self, input_dim, units, mu, sigma, w_init = tf.random_normal_initializer(), mean_isTrainable=True, variance_isTrainable=True, weight_trainable=True, data_type='float32'):
        """Constructor.

        Args:
            input_dim: dimension of input.
            units: the number of neurons.
            mu: the array of expected values.
            sigma: the array of variance.
            w_init: initial weights.
            mean_isTrainable: True if mean is trainable.
            variance_isTrainable: True is variance is trainable.
            weight_trainable: True if weight is trainable.
        """
        super(Gaussian, self).__init__()
        self.data_type = data_type
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype=data_type),
            trainable=weight_trainable,
            name = 'w'
        ) 
        self.mu = tf.Variable(initial_value=mu, trainable=mean_isTrainable, dtype=data_type, name='mu')
        self.sigma = tf.Variable(initial_value=sigma, trainable=variance_isTrainable, dtype=data_type, name='sigma')
       
     
    def call(self, inputs):
        """Forward-pass action for the rest gaussian layer.
        The layer takes additional input for the corresponding 1-D coordinate component.

        Args:
            inputs: output of the previous layer.
            coord_comp: the coordinate component x in f=e^[-0.5*(x-mu)^2/sigma]. 
        Return:
            output of the Gaussian layer.
        """
        input, coord_comp = inputs
        return  tf.reduce_sum(self.w * tf.expand_dims(input,axis=-1), axis=1) * tf.exp(tf.constant(-0.5,dtype=self.data_type)*(coord_comp-self.mu)**2/self.sigma**2)

#Utility
#Todo (SX) Give more options to the initialization of mean and variance arrays. 
def cmptGaussCenterandWidth(lb_arr, ub_arr, N_arr, sigma_mode="identical", data_type='float32'):
    """Compute the array of initial centers and widths by the lower and upper bounds, and grid_size (N_arr) of N dimensions
        By default, mean values will be evenly distributed between lower and upper bounds. variance will be the step.
        Arg:
            lb_arr: array of lower bounds
            ub_arr: array of upper bounds
            N_arr:  array of grids
    """
    mu_arr_list, sigma_arr_list = [], []
   
    for (lb, ub, N) in zip(lb_arr, ub_arr, N_arr):
        mu = tf.cast(tf.linspace(lb,ub,N),dtype=data_type) # the lb and ub will be included.
        if sigma_mode == "identical":
            sigma = tf.constant(mu[1]-mu[0],dtype=data_type) #sigma = 1 step all neurons of a layer
        elif sigma_mode =="distinct":
            sigma = (mu[1]-mu[0])*tf.ones(N,dtype=data_type) #sigma = 1 step per neuron of a layer
        else:
            raise Exception('Unknown sigma mode.')

        mu_arr_list.append(mu)
        sigma_arr_list.append(sigma)
    
    return mu_arr_list, sigma_arr_list 