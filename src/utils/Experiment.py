import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from datetime import datetime as dt
import os, json

class Experiment:

    def __init__(self, model, optimizer, exptConfig):

        self.now        = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
        self.exptConfig = exptConfig
        self.model      = model
        self.optimizer  = optimizer
        
        self.exptFolder = os.path.join( exptConfig['OtherParams']['exptBaseFolder'], self.now )

        # # Save the entire model
        # # ------------------------
        # modelFolder = os.path.join(self.exptFolder, 'model')
        # os.makedirs( modelFolder )
        # self.model.save( modelFolder )

        # All the logs go here ...
        # ------------------------

        self.createMetaData()
        
        self.logDir       = os.path.join(exptConfig['OtherParams']['exptBaseFolder'], self.now)
        self.scalarWriter = tf.summary.create_file_writer( os.path.join( self.logDir, 'scalars', 'metrics' ) )
        self.graphWriter  = tf.summary.create_file_writer( os.path.join( self.logDir, 'graph' ) )


        # Generate a model graph 
        # ---------------------------
        

        return

    def step(self, x, y, stepNumber=0):

        with tf.GradientTape() as tape:

            yHat  = self.model.call(x)
            loss  = tf.reduce_mean((y - yHat)**2)

            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip( grads, self.model.trainable_weights ))

        with self.scalarWriter.as_default():
            tf.summary.scalar('training loss', data=loss, step=stepNumber)

        return loss.numpy()

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


