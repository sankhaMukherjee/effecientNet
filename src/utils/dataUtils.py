import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def getMNISTData(reshape=True):

    # Create the data
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = y_train.astype('float32')
    y_test  = y_test.astype('float32')

    if reshape:
        x_train = x_train.reshape(-1, 784).astype('float32')
        x_test  = x_test.reshape(-1, 784).astype('float32')

    return (x_train, y_train), (x_test, y_test)
    
# ----------------------------------------------------------------------
# Imagenette2-160
# ----------------------------------------------------------------------

def generateImageNette(folder, newFolder):

    builder = tfds.folder_dataset.ImageFolder(root_dir = folder)
    print(builder.info)

    os.makedirs( newFolder, exist_ok=True )

    # ------------------------------------------------
    # Generate training data 
    # ------------------------------------------------
    trainDataX = []
    trainDataY = []

    trainDataset = builder.as_dataset(split='train')
    for d in tqdm(trainDataset, total = 9469):

        im = d['image'].numpy().astype('float32')/255
        im = im[:160, :160, :]
        label = d['label'].numpy().astype('float32')

        trainDataX.append(im)
        trainDataY.append(label)

    trainDataX = np.array(trainDataX)
    trainDataY = np.array(trainDataY)

    with open(os.path.join(newFolder, 'trainX.npy'), 'wb') as fOut:
        np.save( fOut, trainDataX)
    with open(os.path.join(newFolder, 'trainY.npy'), 'wb') as fOut:
        np.save( fOut, trainDataY)

    # ------------------------------------------------
    # Generate testing data 
    # ------------------------------------------------
    trainDataX = []
    trainDataY = []

    trainDataset = builder.as_dataset(split='val')
    for d in tqdm(trainDataset, total = 3925):

        im = d['image'].numpy().astype('float32')/255
        im = im[:160, :160, :]
        label = d['label'].numpy().astype('float32')

        trainDataX.append(im)
        trainDataY.append(label)

    trainDataX = np.array(trainDataX)
    trainDataY = np.array(trainDataY)

    print(trainDataY.shape)

    with open( os.path.join(newFolder, 'testX.npy'), 'wb' ) as fOut:
        np.save( fOut, trainDataX)
    with open( os.path.join(newFolder, 'testY.npy'), 'wb' ) as fOut:
        np.save( fOut, trainDataY)

        
    return

def convertLabelsToOHE(original, categories='auto'):

    ohe    = OneHotEncoder(categories=categories)
    ohe.fit(original.reshape(-1,1))
    result = ohe.transform(original.reshape(-1,1)).toarray()
    
    return result 

def convertFilesToOHE(folder, fileName, categories):

    data = np.load(os.path.join(folder, fileName))
    data = convertLabelsToOHE( data, categories )

    newFileName = fileName.replace('.npy', '_OHE.npy')
    with open( os.path.join(folder, newFileName), 'wb') as fOut:
        np.save(fOut, data)

    return

def getImageNetteData():

    folder = '/home/sankha/Documents/mnt/hdd01/data/imagenette2-160-160'
    
    x_train = np.load( os.path.join(folder, 'trainX.npy') )
    x_test = np.load( os.path.join(folder, 'testX.npy') )

    y_train = np.load( os.path.join(folder, 'trainY_OHE.npy') )
    y_test = np.load( os.path.join(folder, 'testY_OHE.npy') )

    return (x_train, y_train), (x_test, y_test)

def main():

    folder    = '/home/sankha/Documents/mnt/hdd01/data/imagenette2-160'
    folderNew = folder + '-160'
    # generateImageNette(folder, folderNew)

    categories = [list(range(10))]
    convertFilesToOHE( folderNew, 'trainY.npy', categories )
    convertFilesToOHE( folderNew, 'testY.npy', categories )

    return

if __name__ == "__main__":
    main()