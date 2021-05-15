import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from tensorflow.keras.losses  import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

from datetime import datetime as dt
import os, json
import numpy as np

class ExperimentClassify:

    def __init__(self, model, optimizer, exptConfig):

        self.now        = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
        self.exptConfig = exptConfig
        self.model      = model
        self.optimizer  = optimizer
        self.loss       = CategoricalCrossentropy(from_logits=exptConfig['LossParams']['fromLogits'])
        
        # ------------ metrics ----------------------
        self.catAccTest  = CategoricalAccuracy()
        self.catAccTrain = CategoricalAccuracy()

        self.exptFolder  = os.path.join( exptConfig['OtherParams']['exptBaseFolder'], self.now )
        self.modelFolder = os.path.join( self.exptFolder, 'model' )
        self.chkptFolder = os.path.join( self.exptFolder, 'checkpoints' )

        os.makedirs( self.modelFolder, exist_ok=True )
        os.makedirs( self.chkptFolder, exist_ok=True )

        self.stepNumber  = 0
        self.evalNumber  = 0
        self.epoch       = 0

        # All the logs go here ...
        # ------------------------
        self.createMetaData()
        
        self.logDir       = os.path.join(self.exptFolder, 'logs')
        self.scalarWriter = tf.summary.create_file_writer( os.path.join( self.logDir, 'scalars', 'metrics' ) )
        self.graphWriter  = tf.summary.create_file_writer( os.path.join( self.logDir, 'graph' ) )

        return

    def step(self, x, y):

        with tf.GradientTape() as tape:

            yHat  = self.model.call(x)
            loss  = self.loss(y, yHat)

            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip( grads, self.model.trainable_weights ))
        
        self.catAccTrain.update_state(y, yHat)

        with self.scalarWriter.as_default():
            tf.summary.scalar('training loss',     data=loss, step=self.stepNumber)
            tf.summary.scalar('training accuracy', data= self.catAccTrain.result().numpy(), step=self.stepNumber)

        self.stepNumber += 1

        return loss.numpy()

    def eval(self, x, y):

        yHat = self.model.predict(x)
        self.catAccTest.update_state(y, yHat)

        with self.scalarWriter.as_default():
            tf.summary.scalar('testing accuracy', data= self.catAccTest.result().numpy(), step=self.evalNumber)

        self.evalNumber += 1

        return self.catAccTest.result().numpy()

    def createMetaData(self):

        if not os.path.exists(self.exptFolder):
            os.makedirs( self.exptFolder )

        with open( os.path.join(self.exptFolder, 'config.json'), 'w' ) as fOut:
            json.dump( self.exptConfig, fOut )

        return

    def createModelSummary(self, x):
        tf.summary.trace_on(graph=True)
        self.model.predict(x)
        with self.graphWriter.as_default():
            tf.summary.trace_export('name', step=0)
        tf.summary.trace_off()

    def saveModel(self):

        try:
            self.model.save( self.modelFolder )
        except Exception as e:
            print(f'Unable to save the model: {e}')

        return

    def checkPoint(self):
        try:
            epoch = self.epoch
            step = self.stepNumber
            self.model.save_weights( os.path.join( self.chkptFolder, f'{epoch:07d}-{step:07d}' ) )
        except Exception as e:
            print(f'Unable to checkpoint: {self.stepNumber}: {e}')
        return


