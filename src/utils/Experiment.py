import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from datetime import datetime as dt
import os

class Experiment:

    def __init__(self, model, optimizer, logFolderBase='../logs'):

        self.now       = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
        self.model     = model
        self.optimizer = optimizer

        # All the logs go here ...
        # ------------------------
        # self.logFolderBase = logFolderBase
        # self.logDir        = os.path.join(logFolderBase, 'scalars', self.now)

        # self.fileWriter    = tf.summary.create_file_writer( os.path.join( self.logDir, 'metrics' ) )

        return

    def step(self, x, y, stepNumber=0):

        with tf.GradientTape() as tape:

            yHat  = self.model.call(x)
            loss  = tf.reduce_mean((y - yHat)**2)

            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip( grads, self.model.trainable_weights ))

        # with self.fileWriter.as_default()
        #     tf.summary.scalar('training loss', data=learning_rate, step=stepNumber)

        return loss.numpy()


