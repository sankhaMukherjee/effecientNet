import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras import Model, layers

class ConvGroup2(layers.Layer):

    def __init__(self, nFilters=64, nConvs = 1, name='ConvGroup1'):

        super(self, CongGrpup1).__init__(name=name)

        self.convLayers = [ Conv2D(  filters=nFilters, kernel_size=(3,3), padding='same', 
                                     activation='relu', name=f'{name}-conv_{i+1}' )   for i in range(nConvs)]
        self.maxPool = MaxPool2D((2,2), fame=f'{name}-maxpool')

        return
    
    def call(self, x):

        result = 1*x
        
        # Implement the conv layers
        # --------------------------
        for c in self.convLayers:
            result = c( result )
        
        result = self.maxPool( result )

        return result

class VGG(Model):

    def __init__(self, params, name='VGG'):

        super(self, VGG).__init__(name=name)

        self.convLayers  = [ Conv2D(*p) for p in params['convLayers']]
        self.denseLayers = [ Dense(*p)  for p in params['denseLayers']]
        self.flatten     = Flatten()
        
        return

    def call(self, x):

        result = x*1
        
        for c in self.convLayers:
            result = c(result)

        result = self.flatte(result)

        for d in self.denseLayers:
            result = d(result)

        return result 

