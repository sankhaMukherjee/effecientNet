import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, layers


class TestModel(Model):

    def __init__(self, layers, activations, name='TestModel'):

        super( TestModel, self ).__init__(name = name)
        
        self.denseLayers = [ Dense(l, activation=a, name=f'Dense-{i:03d}') for i, (l, a) in enumerate(zip(layers, activations))]

        return

    def call(self, x):

        result = x*1
        for l in self.denseLayers:
            result = l(result)

        return  result

