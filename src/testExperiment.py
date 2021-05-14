import json, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from utils.Experiment             import Experiment
from utils                        import dataUtils  as dU
from models.TF.TestModel          import TestModel
from tensorflow.keras.optimizers  import Adam


def main():

    # ------------- Generate the Parameters --------------------------
    baseConfigs     = json.load(open('configs/testModelConfig.json'))
    modelParams     = baseConfigs['ModelParams']
    optimizerParams = baseConfigs['OptimizerParams']
    otherParams     = baseConfigs['OtherParams']

    # ------------- Generate an Experiment --------------------------
    model = TestModel( **modelParams )
    opt   = Adam( **optimizerParams )
    exp   = Experiment( model, opt )

    # ------------- Generate the Data -------------------------------
    (x_train, y_train), (x_test, y_test) = dU.getMNISTData()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=2048).batch(otherParams['BATCHSIZE'])

    # ------------- Run the Experiment -------------------------------
    for epoch in range( otherParams['EPOCHS'] ):

        for step, (x, y) in enumerate(train_dataset):
            loss = exp.step(x, y)

            if step % otherParams['printEvery'] == 0:
                print(f'{epoch:05d} | {step:05d} | {loss:10.4e} ')

        print(f'\n {epoch:05d} | {step:05d} | {loss:10.4e} \n')

    return

if __name__ == "__main__":
    main()
