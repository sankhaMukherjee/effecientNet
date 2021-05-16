import json, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

from utils.ExperimentClassify     import ExperimentClassify
from utils                        import dataUtils  as dU
from models.TF.VGG                import VGG
from tensorflow.keras.optimizers  import Adam
from tqdm                         import tqdm

def main():

    configFiles = {
        "VGG-11" : 'configs/vgg11Params.json',
        "VGG-13" : 'configs/vgg13Params.json',
        "VGG-16" : 'configs/vgg16Params.json',
        "VGG-19" : 'configs/vgg19Params.json',
    }

    result = []
    for name, configFile in configFiles.items():
        name, trainAcc, maxTrainAccs, testAcc, maxTestAccs = runVGGModel( name, configFile )
        result.append(f'{name} | {trainAcc*100:7.3f} | {maxTrainAccs*100:7.3f} | {testAcc*100:7.3f} | {maxTrainAccs*100:7.3f}')

    result = '\n'.join( result )
    print(result)


    return

def runVGGModel(name, configFile):

    print(f'+----------------------------------------------------------------')
    print(f'| {name} ')
    print(f'+----------------------------------------------------------------')

    # ------------- Generate the Parameters --------------------------
    print('------------- [Generating the basic parameters] --------------')
    baseConfigs     = json.load(open(configFile))
    modelParams     = baseConfigs['ModelParams']
    optimizerParams = baseConfigs['OptimizerParams']
    otherParams     = baseConfigs['OtherParams']

    # ------------- Generate the Data -------------------------------
    print('------------- [Preparing the data] ---------------------------')
    (x_train, y_train), (x_test, y_test) = dU.getImageNetteData()
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(otherParams['BATCHSIZE'])    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(otherParams['BATCHSIZE'])    

    # ------------- Generate an Experiment --------------------------
    print('------------- [Preparing the Experiment] ---------------------')
    model = VGG( modelParams, name=modelParams['name'] )
    opt   = Adam( **optimizerParams )
    exp   = ExperimentClassify( model, opt, baseConfigs )

    # ------------- Save the graph -------------------------------
    print('------------- [Creating the model summary] ------------------')
    exp.createModelSummary( x_test[:1,:,:,:] )

    # ------------- Run the Experiment ---------------------------

    trainAccs, testAccs = [], []
    print('------------- [Starting a run] -----------------------------')
    for epoch in range( otherParams['EPOCHS'] ):

        exp.catAccTrain.reset_states()
        exp.epoch       = epoch
        # exp.stepNumber  = 0

        
        for step, (x, y) in enumerate(tqdm(train_dataset)):

            loss = exp.step(x, y)

            if step % otherParams['printEvery'] == 0:
                trainAcc = exp.catAccTrain.result().numpy()
                tqdm.write(f'{epoch:05d} | {step:05d} | {loss:10.4e} | {trainAcc:10.04e}')
                trainAccs.append(trainAcc)
                
            # if step % otherParams['chkptEvery'] == 0:
            #     exp.checkPoint()

        exp.catAccTest.reset_states()
        for xt, yt in tqdm( test_dataset ):
            testAcc = exp.eval(xt, yt)
        
        testAccs.append(testAccs)
        print(f'{epoch:05d} | {step:05d} | {loss:10.4e} | {trainAcc:10.04e} | {testAcc:10.04e}')
    
    # ------------- Save the model ---------------------------
    # exp.saveModel()

    return name, trainAcc, max(trainAccs), testAcc, max(testAccs)

if __name__ == "__main__":
    main()
