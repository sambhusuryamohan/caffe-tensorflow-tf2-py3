import tensorflow as tf
import numpy as np

class ParamLoaderFactory:
    @staticmethod
    def create(layer):
        #print("Get Layer", layer.name, type(layer))
        if isinstance(layer, tf.keras.layers.Conv2D):
            return ConvParameter(layer)
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
        if param_name == 'weights':
            self.layer.kernel.assign(data)
        elif param_name == 'biases':
            self.layer.bias.assign(data)
        else:
            raise ValueError('Conv parameter not identified {}'.format(param_name))
        

class BNParameter:
    def __init__(self, layer):
        self.layer = layer

    def load(self, param_name, data):
        if param_name == 'scale':
            self.layer.gamma.assign(data)
        elif param_name == 'offset':
            self.layer.beta.assign(data)
        elif param_name == 'mean':
            self.layer.moving_mean.assign(data)
        elif param_name == 'variance':
            self.layer.moving_variance.assign(data)
        else:
            raise ValueError('BN parameter not identified {}'.format(param_name))


class PReLUParameter:
    def __init__(self, layer):
        self.layer = layer

    def load(self, param_name, data):
        if param_name == 'alpha':
            self.layer.alpha.assign(np.broadcast_to(data, self.layer.alpha.shape))
        else:
            raise ValueError('PReLU parameter not identified {}'.format(param_name))

class DenseParameter:
    def __init__(self, layer):
        self.layer = layer

    def load(self, param_name, data):
        if param_name == 'weights':
            self.layer.kernel.assign(data)
        elif param_name == 'biases':
            self.layer.bias.assign(data)
        else:
            raise ValueError('Dense parameter not identified |{}|'.format(param_name))
