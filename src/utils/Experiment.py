import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from datetime import datetime as dt
import os, json
import numpy as np

class Experiment:

    def __init__(self, model, optimizer, exptConfig):

        self.now        = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
        self.exptConfig = exptConfig
        self.model      = model
        self.optimizer  = optimizer
        
        self.exptFolder  = os.path.join( exptConfig['OtherParams']['exptBaseFolder'], self.now )
        self.modelFolder = os.path.join( self.exptFolder, 'model' )
        self.chkptFolder = os.path.join( self.exptFolder, 'checkpoints' )

        os.makedirs( self.modelFolder, exist_ok=True )
        os.makedirs( self.chkptFolder, exist_ok=True )

        self.stepNumber  = 0
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
            loss  = tf.reduce_mean((y - yHat)**2)

            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip( grads, self.model.trainable_weights ))

        with self.scalarWriter.as_default():
            tf.summary.scalar('training loss', data=loss, step=self.stepNumber)

        self.stepNumber += 1

        return loss.numpy()

    def eval(self, x, y):

        yHat = self.model.predict(x)
        loss = np.mean((y - yHat)**2)

        with self.scalarWriter.as_default():
            tf.summary.scalar('testing loss', data=loss, step=self.stepNumber)

        return loss

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


