# import the necessary packages
import tensorflow as tf
import numpy as np

import sys
from packaging import version

if version.parse(tf.__version__) < version.parse("2.0"):
    from keras import layers, Model
else:
    from tensorflow.keras import layers, Model

class DenseBlock(layers.Layer):
    """A Dense block of convolutional layers. 
    
    This function creates a stack of 'n_layers' convolutional layers, 
    where each layer's input is not just the previous layer output, but a concatenation of all
    previous layers' outputs.

    Every layers uses 'k_factor' filters (kernels) with shape 'conv_filter'

    The type of activation uses in every layer can be either 'relu' or 'lrelu' by setting argument 'activations'

    Argument 'batch_norm' tells if batch_norm layer should be used after 
       
    """

    def __init__(self, conv_filter, n_layers=4, k_factor=4, activations='relu', batch_norm=True, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        
        if activations != 'relu' and activations != 'lrelu':
            raise ValueError ("Non-supported activation type '{}'".format(activations))
        
        self.conv_filter = conv_filter
        self.n_layers = int(n_layers)
        self.k_factor = k_factor
        self.activations = activations
        self.batch_norm = batch_norm
        
    def build(self, input_shape):
        self.conv_layers = [layers.Conv2D(self.k_factor, self.conv_filter, activation=None, padding='same', trainable=True) for _ in range(self.n_layers)]
        
        if self.batch_norm:
          self.batch_norm_layers = [layers.BatchNormalization(trainable=True) for _ in range(self.n_layers - 1)]
 
    def call(self, inputs):
        x = inputs
        for i, conv_layer in enumerate(self.conv_layers):
            y = conv_layer(x)
            
            if self.activations == 'relu':
              y = tf.nn.relu(y)
            if self.activations == 'lrelu':
              y = tf.nn.leaky_relu(y, alpha=0.3)
            
            if i < (self.n_layers - 1):
                x = tf.concat([y, x], axis=-1)
                if self.batch_norm:
                    x = self.batch_norm_layers[i](x)
        return y

    def get_config(self):
        config = super(DenseBlock, self).get_config()
        config.update({"conv_filter": self.conv_filter,
                        "n_layers" : self.n_layers,
                        "k_factor" : self.k_factor,
                        "activations": self.activations, 
                        "batch_norm": self.batch_norm})
        return config

class DenseConvStack(layers.Layer):
    """Creates a stack of DenseBlock layers. After each Denseblock a max-pool layer
    of 'max_pool_stride' is applied.
    
    The depth of each Denseblock can be controlled by argument 'densenet_depth', while
    the depth of their (stack + pool) is controlled using 'conv_depth' argument.

    Instead of using just a single type of filters for every convolution layer, a list
    of tuples with different kernel shapes can be used to create paralelal conv stacks.
    Such list is the argument 'kernel_sshape_list'

    Also, the number of each type of filter that will be applied at every stage can be set
    by changing argument 'k_factor_list'.
 
    Args:
        layers ([type]): [description]
    """
    def __init__(self,
                 conv_depth=4,                                  # number of convolutional layers
                 kernel_shape_list=(3,3),                       # a list of tuples with the kernel shapes to be used in the convNets. Same length as k_factor_list
                 k_factor_list=8,                               # a list of ints with the k_factor to be used in the DenseNets. Same length as kernel_shape_list
                 use_exp_channels=False,                        # whether we should increase the number of channels (k_factors) exponentially(**2) in each downsampleed DenseBlock
                 densenet_depth=4,                              # number of layers that each dense block is going to have
                 max_pool_stride=[1,2,2,1],                     # kernel shape and stride to use for max pool. Should be 4D like [1,2,2,1]. Use None or [1,1,1,1] to not perform max pool
                 activations='relu',                            # types of activation function to be used inside the model
                 batch_norm={'densenet':True, 'concat': True},  # whether or not to use batch norm inside every DenseBlock or only after the concatenationg and feature-maps pulling
                 dropout={'concat': 0.25},                      # whether or not to usse dropout
                 **kwargs):
        
        super(DenseConvStack, self).__init__(**kwargs)
        
        # Setting up 'kernel_shape_list' argument as always a list
        if type(kernel_shape_list) == tuple:
            if len(kernel_shape_list) == 2:
                kernel_shape_list = [kernel_shape_list]
            else:
                raise ValueError('The tuple used as \'kernel_shape_list\' must be 2D')
        
        elif type(kernel_shape_list) == int:
            kernel_shape_list = [(kernel_shape_list,1), (1, kernel_shape_list)]

        elif type(kernel_shape_list) == list:    
            if type(kernel_shape_list[0]) == list:
                for i,k in enumerate(kernel_shape_list):
                    assert len(k) == 2, "Make sure there are only 2D tuples or lists in \'kernel_shape_list\'"
                    if type(k) is list:
                        k = tuple(k)
                        kernel_shape_list[i] = k
                    else:
                        raise ValueError ('\'kernel_shape_list\' must be a list of 2D tuples or 2D lists.')
            elif len(kernel_shape_list) == 1:
                if type(kernel_shape_list[0]) == int:
                    kernel_shape_list = [(kernel_shape_list[0], 1), (1, kernel_shape_list[0])]
                elif type(kernel_shape_list[0]) == tuple:
                    assert len(kernel_shape_list[0]) == 2, "Make sure there are only 2D tuples in \'kernel_shape_list\'"
                else:
                    raise ValueError('\'kernel_shape_list\' must be a list of 2D tuples or a single integer number')
            else:
                for kernel in kernel_shape_list:
                    #assert len(kernel) == 2, "\'kernel_shape_list\' must be a list of 2D tuples or a single integer number"
                    assert (type(kernel) == tuple) and (len(kernel) == 2), "\'kernel_shape_list\' must be a list of 2D tuples or a single integer number"  
        
        # if a single int was used as argument, copy it into a list with the correct length
        if type(k_factor_list) is int:
            k_factor_list = [k_factor_list]*len(kernel_shape_list)
        elif len(k_factor_list) != len(kernel_shape_list):
            raise ValueError('k_factor_list must be a single integer or a list of integers with the same length as kernel_shape_list') 
        
        if dropout['concat'] is not None and dropout['concat'] != 0:
            assert (dropout['concat'] > 0 and dropout['concat'] < 1), "Invalid Dropout 'concat' Value. It should be a number between 0 and 1 or None."
                    
        self.conv_depth = conv_depth                         # number of convolutional layers
        self.kernel_shape_list = kernel_shape_list           # a list of tuples with the kernel shapes to be used in the convNets. Same length as k_factor_list
        self.k_factor_list = k_factor_list                   # a list of ints with the k_factor to be used in the DenseNets. Same length as kernel_shape_list
        self.n_branches = len(self.kernel_shape_list)
        self.use_exp_channels= use_exp_channels              # whether we should increase the number of channels exponentially(**2) in each downsample DenseLayer
        self.densenet_depth = densenet_depth                 # number of layers that each dense block is going to have
        self.max_pool_stride = max_pool_stride               # kernel shape and stride to use for max pool. Should be 4D like [1,2,2,1]. Use None or [1,1,1,1] to not perform max pool
        self.activations = activations                       # types of activation function to be used inside the model
        self.batch_norm = batch_norm
        self.dropout = dropout
        
    def build(self, input_shape):
        if self.use_exp_channels:
          self.dense_blocks = [DenseBlock(conv_filter=self.kernel_shape_list[i],
                                           n_layers=self.densenet_depth, 
                                           k_factor=self.k_factor_list[i]*(2**depth), 
                                           activations=self.activations, 
                                           batch_norm=self.batch_norm['densenet'],
                                           name='dense_block_branch{}_depth{}'.format(i, depth),
                                           trainable=True) for depth in range(self.conv_depth) for i in range(self.n_branches)]
        else:
          self.dense_blocks = [DenseBlock(conv_filter=self.kernel_shape_list[i],
                                           n_layers=self.densenet_depth, 
                                           k_factor=self.k_factor_list[i], 
                                           activations=self.activations, 
                                           batch_norm=self.batch_norm['densenet'],
                                           name='dense_block_branch{}_depth{}'.format(i, depth),
                                           trainable=True) for depth in range(self.conv_depth) for i in range(self.n_branches)]
        if self.batch_norm['concat']:
          self.batch_norm_layers = [layers.BatchNormalization(trainable=True) for _ in range(self.conv_depth - 1)]
    
    
    def call(self, inputs, training=None):
        x = inputs
        for depth in range(self.conv_depth):
            b = [0]*self.n_branches
            for i in range(self.n_branches):
                y = self.dense_blocks[(depth*self.n_branches) + i](x)   
                # Max pooling to downsample
                if self.max_pool_stride == [1,1,1,1] or self.max_pool_stride is None:
                    b[i] = b
                else:
                    b[i] = tf.nn.max_pool2d(y,  ksize=self.max_pool_stride, strides=self.max_pool_stride, padding='VALID')
            x = tf.concat(b, axis=-1)   
            
            if depth != (self.conv_depth - 1):
                x = self.batch_norm_layers[depth](x) if self.batch_norm['concat'] else x
                
                if depth != 0:
                    if training:
                        x = tf.nn.dropout(x, rate=self.dropout['concat'])
        return x

    def get_config(self):
        config = super(DenseConvStack, self).get_config()
        config.update({"conv_depth"       : self.conv_depth, 
                       "kernel_shape_list": self.kernel_shape_list, 
                       "k_factor_list"    : self.k_factor_list, 
                       "use_exp_channels" : self.use_exp_channels, 
                       "densenet_depth"   : self.densenet_depth, 
                       "max_pool_stride"  : self.max_pool_stride,
                       "activations"      : self.activations,  
                       "batch_norm"       : self.batch_norm,
                       "dropout"          : self.dropout})
        return config


class DenseMultiKernelConvMLP(Model):
    """This class creates a VGG-ish classifier, but using DenseNets instead of regular stack of conv layers.
    First, a DenseConvStack is created and its output is sent to 'fc_depth' fully connected layers.

    The last fully connected layer has 'n_classes' neurons.
    """
    def __init__(self,
                 n_classes=11, 
                 conv_depth=4, 
                 kernel_shape_list=(3,3), 
                 k_factor_list=8, 
                 use_exp_channels=False, 
                 densenet_depth=4,
                 max_pool_stride=[1,2,2,1], 
                 activations='lrelu',  
                 final_activation='sigmoid',
                 fc_depth=3,
                 batch_norm={'densenet':False, 'concat': False, 'flatten': False, 'fc_layers': False},
                 dropout={'concat': 0.25, 'flatten': 0.25, 'fc_layers': 0.25},
                 **kwargs):
        
        super(DenseMultiKernelConvMLP, self).__init__(**kwargs)
        
        
        if dropout['flatten'] is not None and dropout['flatten'] != 0:
            assert (dropout['flatten'] > 0 and dropout['flatten'] < 1), "Invalid Dropout 'flatten' Value. It should be a number between 0 and 1 or None."
        
        if dropout['fc_layers'] is not None and dropout['fc_layers'] != 0:
            assert (dropout['fc_layers'] > 0 and dropout['fc_layers'] < 1), "Invalid Dropout 'fc_layers' Value. It should be a number between 0 and 1 or None."
            
        self.n_classes = n_classes
        self.kernel_shape_list = kernel_shape_list           # a list of tuples with the kernel shapes to be used in the convNets. Same length as k_factor_list
        self.k_factor_list = k_factor_list                   # a list of ints with the k_factor to be used in the DenseNets. Same length as conv_filter_list
        self.use_exp_channels = use_exp_channels
        self.densenet_depth = densenet_depth
        self.max_pool_stride = max_pool_stride
        self.activations = activations
        self.final_activation = final_activation
        self.fc_depth = fc_depth
        self.batch_norm = batch_norm
        self.dropout = dropout
        
        self.input_names = ['main_input']
        self.n_inputs = len(self.input_names) 
        self.n_channels = 1

        self.flatten = layers.Flatten(trainable=False)
        self.conv_stack = [DenseConvStack(conv_depth=conv_depth, 
                                           kernel_shape_list=kernel_shape_list, 
                                           k_factor_list=k_factor_list, 
                                           use_exp_channels=use_exp_channels, 
                                           densenet_depth=densenet_depth,
                                           max_pool_stride=max_pool_stride, 
                                           activations=self.activations,  
                                           batch_norm=self.batch_norm,
                                           dropout=self.dropout,
                                           name=input_names + '_branch',
                                           trainable=True,
                                           **kwargs) for input_names in self.input_names]
        
    def build(self, input_shape):    
        self.hidden_connected = [layers.Dense(2**(6-depth), 
                                              activation=None, 
                                              name='dense_hidden_layer{}'.format(depth),
                                              trainable=True) for depth in range(self.fc_depth - 1)]
        
        self.last_layer = layers.Dense(self.n_classes, 
                                       activation=self.final_activation, 
                                       name='output',
                                       trainable=True) 
        if self.batch_norm['flatten']:
            self.batch_norm_flatten = layers.BatchNormalization(trainable=True)
        if self.batch_norm['fc_layers']:
            self.batch_norm_fc_layers = [layers.BatchNormalization(trainable=True) for _ in range(self.fc_depth-1)]
    
    def call(self, inputs, training=None):
        if self.n_inputs == 1:
            x = self.conv_stack[0](inputs)
        else:
            x = self.conv_stack[0](inputs[0])
            y = self.conv_stack[1](inputs[1])
            x = tf.concat([x,y], axis=-1)
        
        x = self.flatten(x)
        
        if self.batch_norm['flatten']:
            x = self.batch_norm_flatten(x)

        if training:
            if self.dropout['flatten'] is not None and self.dropout['flatten'] != 0:
                x = tf.nn.dropout(x, rate=self.dropout['flatten'])
        
        for depth in range(self.fc_depth - 1):
            x = self.hidden_connected[depth](x)
            
            if self.activations == 'relu':
              x = tf.nn.relu(x)
            if self.activations == 'lrelu':
              x = tf.nn.leaky_relu(x, alpha=0.3)
            
            if self.batch_norm['fc_layers']:
                x = self.batch_norm_fc_layers[depth](x)
            
            if self.dropout['fc_layers'] is not None and self.dropout['fc_layers'] != 0:
                x = tf.nn.dropout(x, rate=self.dropout['fc_layers'])(x)
            
        x = self.last_layer(x)
        return x  
    
    def get_config(self):
        config = super(DenseMultiKernelConvMLP, self).get_config()
        config.update({"n_classes"        : self.n_classes,
                       "conv_depth"       : self.conv_stack[0].conv_depth, 
                       "kernel_shape_list": self.kernel_shape_list, 
                       "k_factor_list"    : self.k_factor_list, 
                       "use_exp_channels" : self.use_exp_channels, 
                       "densenet_depth"   : self.densenet_depth, 
                       "max_pool_stride"  : self.max_pool_stride,
                       "activations"      : self.activations,
                       "final_activation" : self.final_activation,
                       "fc_depth"         : self.fc_depth,
                       "batch_norm"       : self.batch_norm,
                       "dropout"          : self.dropout})
        return config

def buildComplexGenreClassifier(input_shape, 
                         n_classes=11, 
                         conv_depth=4, 
                         kernel_shape_list=(3,3), 
                         k_factor_list=8, 
                         use_exp_channels=False, 
                         densenet_depth=4, 
                         max_pool_stride=[1,2,2,1],
                         activations='relu',  
                         final_activation='softmax',
                         fc_depth=3,
                         batch_norm={'densenet':True, 'concat': True, 'flatten': True, 'fc_layers': True},
                         dropout={'concat': 0.0, 'flatten': 0.2, 'fc_layers': 0.2}, **kwargs):

    """This function creates a VGG-ish classifier, but using DenseNets instead of regular stack of conv layers.
    First, a DenseConvStack is created and its output is sent to 'fc_depth' fully connected layers.

    The last fully connected layer has 'n_classes' neurons.
    """
    
    input_layer = layers.Input(shape=input_shape, name='main_input')
    inputs=[input_layer]
    x = DenseConvStack(conv_depth=conv_depth, 
                    kernel_shape_list=kernel_shape_list, 
                    k_factor_list=k_factor_list, 
                    use_exp_channels=use_exp_channels, 
                    densenet_depth=densenet_depth,
                    max_pool_stride=max_pool_stride, 
                    activations=activations,  
                    batch_norm=batch_norm,
                    dropout=dropout,
                    name='main_input_convbranch',
                    **kwargs)(input_layer)
        
    x = layers.Flatten()(x)
    
    if batch_norm['flatten']:
        x = layers.BatchNormalization()(x)
    
    if dropout['flatten'] is not None and dropout['flatten'] != 0:
        assert (dropout['flatten'] > 0 and dropout['flatten'] < 1), "Invalid Dropout Value. It should be a number between 0 and 1 or None."
        x = layers.Dropout(dropout['flatten'])(x)
    
    for depth in range(fc_depth - 1):
        x = layers.Dense(2**(6-depth), activation=None, name='dense_hidden_layer{}'.format(depth))(x) 
        
        if activations == 'relu':
            x = tf.nn.relu(x)
        if activations == 'lrelu':
            x = tf.nn.leaky_relu(x, alpha=0.3)    
        
        if batch_norm['fc_layers']:
            x = layers.BatchNormalization()(x)

        if dropout['fc_layers'] is not None and dropout['fc_layers'] != 0:
            assert (dropout['fc_layers'] > 0 and dropout['fc_layers'] < 1), "Invalid Dropout Value. It should be a number between 0 and 1 or None."
            x = layers.Dropout(dropout['fc_layers'])(x)    
    
    x = layers.Dense(n_classes, activation=final_activation, name='output')(x)

    model = Model(inputs=inputs, outputs=x)
    return model