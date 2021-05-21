# EffecientNet

My own implementation of EffecientNets. 

This implementation is not intended to be a replacement of the original verison of EffecientNet
that is present in the paper, but rather a very simple implementation that is easy to read,
understand and experiment with. 

The idea behind this implementation is that instead of needing to tweaking the codeextensively, 
it should be relatively easy to tweak the code through configuraiton files. This will then
allow for further optimizaiton through Neural Architecture Search.

This is not intended to be production-level code, and no guarantees are made of its reliability.
Use it at your own risk.

# 2. Dataset

This example uses a couple of datasets. They are as folliws:

## 2.1. [MNIST](http://yann.lecun.com/exdb/mnist/)

This is the well-known MNIST dataset. TensorFlow already comes with a dataset, and this is something
you sue directly, without needing to download any data. The dataset will be autimatically downloaded
for you to use when you need it. The file 
[`src/utils/dataUtils.py`](https://github.com/sankhaMukherjee/effecientNet/blob/master/src/utils/dataUtils.py) 
contains a function `getMNISTData()` 
that will allow you to read the data directly. The following transformations are automatically performed:

1. scale the data to the limits [0,1]
2. If needed, reshape the x-values to [-1, 748] so that it can be passed to a Dense network directly.
3. If a classification is required, the y-values can be converted to OHE data directly

## 2.2. [Imagenette](https://github.com/fastai/imagenette)

A smaller version (10 classes) of the popular [Imagenet](https://www.image-net.org/) dataset which
is also present. This is typically used for rapid prototyping on new ideas, since the original
Imagenet dataset is rather big, and takes a long time to tune. The version of dataset used is the
[160 px](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz) imagenette dataset, which
you will have to separately download. To use this dataset, follow the steps below:

### 2.2.1. Download and extract the data

1. go to a favourite folder location: `cd <folder location>`
2. download the data: `wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz`
3. unzip the data: `tar zxvf imagenette2-160.tgz`

### 2.2.2. Normalize the data

The imagenette data comprises of data that has been converted to format such that the shortest side is
160 pixles, and the aspect ratio is maintained. This results in a set of uneven-sized data. For this
case, a simple normalization has been performed - the image is cropped into a 160x160 image. Note that
this is not the ideal way of normalizing the images. However, for this simple set of experiments this
should be sufficient. For normalizing the data, follow the steps:

1. In the `main()` function of the file  [`src/utils/dataUtils.py`](https://github.com/sankhaMukherjee/effecientNet/blob/master/src/utils/dataUtils.py), change the folder to the location of the data where imagenette data is present. 
2. Run this file: `python3 dataUtils.py`
3. A new folder will be created with the name `imagenette2-160-160`
4. In the `getImageNetteData()` function of the same file, update the folder so that it points to this new file.

Now you should be ready to run the examples ...

# Baseline Examples

## VGG architectures

VGG architectures as originally described do not have batch normalization and dropout and hence are prone
to overfitting. Several of the VGG architectures are very similar to one another and can be generated through
the use of a base model defiition 
[src/models/TF/VGG.py](https://github.com/sankhaMukherjee/effecientNet/blob/master/src/models/TF/VGG.py)
and through their respective config files:

 - [src/configs/vgg11Params.json](https://github.com/sankhaMukherjee/effecientNet/blob/master/src/configs/vgg11Params.json)
 - [src/configs/vgg13Params.json](https://github.com/sankhaMukherjee/effecientNet/blob/master/src/configs/vgg13Params.json)
 - [src/configs/vgg16Params.json](https://github.com/sankhaMukherjee/effecientNet/blob/master/src/configs/vgg16Params.json)
 - [src/configs/vgg19Params.json](https://github.com/sankhaMukherjee/effecientNet/blob/master/src/configs/vgg19Params.json)

They can be evaluated with the command (within the `src` folder): `python3 vggExperiments.py`

The following are on the [Imagenette](https://github.com/fastai/imagenette) dataset.

|    model name      | train acc. | max train acc. |  test acc. |
|--------------------|------------|----------------|------------|
|VGG-11              |     99.166 |         99.556 |     67.516 |
|VGG-13              |     99.609 |         99.609 |     64.459 |
|VGG-16              |     98.944 |         99.261 |     54.293 |
|VGG-19              |     98.659 |         98.754 |     57.478 |

Some results are shown below

# Requirements

The current version is written with the following configuration:

 - `CudaToolkit 11.0`
 - `cuDNN 8.`
 - `TensorFlow 2.4.1`
 - `torch 1.8.0+cu11`

The code has been tested on a GPU with the following configuration: 

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 450.119.03   Driver Version: 450.119.03   CUDA Version: 11.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 2070    Off  | 00000000:01:00.0  On |                  N/A |
|  0%   47C    P8    21W / 175W |   1456MiB /  7979MiB |      1%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

For some reason, the current version of tensorflow overflows in memory usage and
errors out for RTX 2070 seres. For that reason, you will need to add the following
lines to your TensorFlow code to prevent that from happening.

```python
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
```

## Authors

Sankha S. Mukherjee - Initial work (2021)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details
