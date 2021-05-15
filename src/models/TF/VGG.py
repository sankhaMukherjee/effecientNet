import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import Model, layers, Sequential

class ConvGroup(layers.Layer):

    def __init__(self, nFilters=64, nConvs = 1, name='ConvGroup'):

        super(ConvGroup, self).__init__(name=name)

        self.convLayers = Sequential([ Conv2D(  filters=nFilters, kernel_size=(3,3), padding='same', activation='relu', ) for i in range(nConvs)])
        self.maxPool    = MaxPool2D((2,2), name=f'{name}-maxpool')
        self.all        = Sequential([ self.convLayers, self.maxPool ], name=name)

        return
    
    def call(self, x):
        return self.all(x)

class VGG(Model):

    def __init__(self, params, name='VGG'):

        super( VGG, self).__init__(name=name)

        self.convLayers  = Sequential([ ConvGroup(**p) for p in params['convLayers']], name='ConvBlocks')
        self.flatten     = Flatten()
        self.denseLayers = Sequential([ Dense(**p)  for p in params['denseLayers']], name='DenseBlocks')
        self.all         = Sequential([ self.convLayers, self.flatten, self.denseLayers ], name=name)
        
        return

    def call(self, x):
        return self.all(x)

