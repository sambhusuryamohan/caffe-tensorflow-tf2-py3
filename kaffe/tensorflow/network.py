import numpy as np
import tensorflow as tf
from .param_loader import ParamLoaderFactory
import json
import os

DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated

class Network(object):

    def __init__(self, inputs, output_node_names, trainable=True):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.setup()
        output_layers = []
        for output_name in output_node_names: 
            output_layers.append(self.layers[output_name])
        self.model = tf.keras.models.Model(inputs=[inputs['data']], outputs=output_layers)
        self.model.trainable=False
        self.model.summary()


    def save(self, saved_model_path): 
        #for i, layer in enumerate(self.model.layers):
        #    print('layer {} weights: {}'.format(i, layer.weights))
        self.model.save(saved_model_path)
        self.model.summary()
        with open(os.path.join(saved_model_path, 'model.json'), 'w') as f:
            json.dump(self.model.to_json(), f)
        self.model.save_weights(os.path.join(saved_model_path, 'model_weights'))


    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')


    def load_parameter(self, layer, param_name, data):
        #print('layer weights', layer.non_trainable_variables)
        param_loader = ParamLoaderFactory.create(layer)
        param_loader.load(param_name, data)

        
    def load(self, data_path, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path, allow_pickle=True).item()
        for op_name in data_dict:
            layer = self.model.get_layer(name=op_name)
            #print("##############{}############".format(op_name))
            for param_name, data in data_dict[op_name].items():
                #print('caffe data', op_name, param_name, np.array(data))
                try:
                    #print("before loading", layer.__dict__)
                    self.load_parameter(layer, param_name, data)
                    #print("after loading", layer.__dict__)
                except ValueError:
                    if not ignore_missing:
                        raise
        data_dict = np.load(data_path, allow_pickle=True).item()
        #for op_name in data_dict:
        #    for param_name, data in data_dict[op_name].items():
        #        print('all caffe data', op_name, param_name, data)

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, str):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=None,
             group=1,
             biased=True):
        # Verify that the padding is acceptable
        #self.validate_padding(padding)'
        # Get the number of channels in the input
        c_i = input.shape[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        scope_name = name[1:] if name[0] == '_' else name 
        activation = None
        if relu:
            activation = 'relu'
        with tf.name_scope(scope_name) as scope:
            #print('conv', scope_name, name, group, k_h, k_w, s_h, s_w, padding, biased, activation)
            if padding != None: 
                input = tf.keras.layers.ZeroPadding2D(padding=padding)(input)
            padding = 'VALID'
            if group == 1:
                output = tf.keras.layers.Conv2D(c_o, kernel_size=(k_h, k_w), strides=(s_h, s_w), use_bias=biased, padding=padding, activation=activation, name=name)(input)

            else:
                #print('depthconv', scope_name, name, group, k_h, k_w, s_h, s_w, padding, biased, activation)
                output = tf.keras.layers.DepthwiseConv2D(kernel_size=(k_h, k_w), strides=(s_h, s_w), padding=padding, use_bias=biased, activation=activation, name=name)(input)
            return output

    @layer
    def relu(self, input, name):
        return tf.keras.layers.ReLU(name=name)(input)

    @layer
    def prelu(self, input, name):
        with tf.name_scope(name):
            output = tf.keras.layers.PReLU(name=name)(input)
        return output

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=None):
        #self.validate_padding(padding)
        if padding != None: 
            input = tf.keras.layers.ZeroPadding2D(padding=padding)(input)
        return tf.keras.layers.MaxPooling2D(
                              pool_size=(k_h, k_w),
                              strides=(s_h, s_w),
                              padding='VALID',
                              name=name)(input)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        #self.validate_padding(padding)
        if padding != None: 
            input = tf.keras.layers.ZeroPadding2D(padding=padding)(input)
        return tf.keras.layers.AveragePooling2D(
                              pool_size=(k_h, k_w),
                              strides=(s_h, s_w),
                              padding='VALID',
                              name=name)(input)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.keras.layers.Lambda(lambda xinput:tf.nn.local_response_normalization(xinput,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name))(input)

    @layer
    def concat(self, inputs, axis, name):
        return tf.keras.layers.Concatenate(axis=axis,  name=name)(inputs)

    @layer
    def add(self, inputs, name):
        scope_name = name[1:] if name[0] == '_' else name
        return tf.keras.layers.Add(name = scope_name)(inputs)

    @layer
    def fc(self, input, num_out, name, relu=True):
        scope_name = name[1:] if name[0] == '_' else name
        activation = None
        with tf.name_scope(scope_name) as scope:
            if relu:
                activation = 'relu'
            flatten = tf.keras.layers.Flatten(name='flatten_'+name)(input)
            fc = tf.keras.layers.Dense(num_out, activation=activation, name=name)(flatten)
            return fc

    @layer
    def softmax(self, input, name):
        return tf.keras.layers.Softmax(name=name)(input)

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False):
        scope_name = name[1:] if name[0] == '_' else name
        with tf.name_scope(scope_name) as scope:
            output = tf.keras.layers.BatchNormalization(
                scale=True,
                epsilon=1e-5,
                name=name)(input)
            if relu:
                output = tf.keras.layers.ReLU(name=name+'/ReLU')(output)
            return output

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.keras.layers.Dropout(keep_prob, name=name)(input)

    '''
    @layer
    def permute(self, input, dim):
        return tf.keras.layers.Permute(dim=axis, name=name)(input)
    '''

    @layer
    def reshape(self, input, shape, name):
        return tf.keras.layers.Reshape(target_shape=shape, name=name)(input)
