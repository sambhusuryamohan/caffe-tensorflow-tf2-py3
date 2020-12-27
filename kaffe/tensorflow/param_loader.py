import tensorflow as tf
import numpy as np

class ParamLoaderFactory:
    @staticmethod
    def create(layer):
        #print("Get Layer", layer.name, type(layer))

        if isinstance(layer, tf.keras.layers.Conv2D):
            if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                params = DWParameter(layer)
            else: 
                params = ConvParameter(layer)
            return params
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            return BNParameter(layer)
        elif isinstance(layer, tf.keras.layers.PReLU):
            return PReLUParameter(layer)
        elif isinstance(layer, tf.keras.layers.Dense):
            return DenseParameter(layer)
        else:
            raise NotImplementedError('layer not implemented {}'.format(type(layer)))
        return None


class ConvParameter:
    def __init__(self, layer):
        self.layer = layer

    def load(self, param_name, data):
        #print('conv', self.layer)
        if param_name == 'weights':
            #shape = self.layer.kernel.shape
            #print('shape', shape)
            #data = data.transpose((1, 2, 0, 3))
            self.layer.kernel.assign(data)
        elif param_name == 'biases':
            #shape = self.layer.bias.shape
            #print('shape', shape)
            #data = data.reshape(shape)
            self.layer.bias.assign(data)
        else:
            raise ValueError('Conv parameter not identified {}'.format(param_name))

class DWParameter:
   def __init__(self, layer):
       self.layer = layer

   def load(self, param_name, data):
       #print('conv', self.layer, data.shape)
       if param_name == 'weights':
           #shape = self.layer.kernel.shape
           #print('shape', shape)
           data = data.transpose((0, 1, 3, 2))
           self.layer.depthwise_kernel.assign(data)
       elif param_name == 'biases':
           #shape = self.layer.bias.shape
           #print('shape', shape)
           #data = data.reshape(shape)
           self.layer.bias.assign(data)
       else:
           raise ValueError('DW parameter not identified {}'.format(param_name))
        

class BNParameter:
    def __init__(self, layer):
        self.layer = layer

    def load(self, param_name, data):
        if param_name == 'scale':
            #shape = self.layer.gamma.shape
            #print('shape', shape)
            #data = data.reshape(shape)
            self.layer.gamma.assign(data)
        elif param_name == 'offset':
            #shape = self.layer.beta.shape
            #print('shape', shape)
            #data = data.reshape(shape)
            self.layer.beta.assign(data)
        elif param_name == 'mean':
            #shape = self.layer.moving_mean.shape
            #print('shape', shape)
            #data = data.reshape(shape)
            self.layer.moving_mean.assign(data)
        elif param_name == 'variance':
            #shape = self.layer.moving_variance.shape
            #print('shape', shape)
            #data = data.reshape(shape)
            self.layer.moving_variance.assign(data)
        else:
            raise ValueError('BN parameter not identified {}'.format(param_name))


class PReLUParameter:
    def __init__(self, layer):
        self.layer = layer

    def load(self, param_name, data):
        if param_name == 'alpha':
            shape = self.layer.alpha.shape
            #print('shape', shape)
            data = np.broadcast_to(data, shape)
            self.layer.alpha.assign(data)
        else:
            raise ValueError('PReLU parameter not identified {}'.format(param_name))

class DenseParameter:
    def __init__(self, layer):
        self.layer = layer

    def load(self, param_name, data):
        if param_name == 'weights':
            #shape = self.layer.kernel.shape
            #print('shape', shape)
            #data = data.transpose((1, 0)).reshape(shape)
            self.layer.kernel.assign(data)
        elif param_name == 'biases':
            #shape = self.layer.bias.shape
            #print('shape', shape)
            #data = data.reshape(shape)
            self.layer.bias.assign(data)
        else:
            raise ValueError('Dense parameter not identified |{}|'.format(param_name))
