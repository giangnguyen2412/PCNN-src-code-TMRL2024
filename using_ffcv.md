# Using FFCV to faster your training speed on ImageNet

## Installation
### Install ```ffcv``` and create a new environment
Follow the [install instructions](https://github.com/libffcv/ffcv/blob/main/README.md) from the authors. This will creat a new virtual environment using conda. 
If you get error: 

```CondaHTTPError: HTTP 403 FORBIDDEN for url https://conda.anaconda.org/pytorch/win-64/pytorch-cpu-1.1.0-py3.7_cpu_1.tar.bz2``` 

when installing ```pytorch``` for the ffcv environment, try to run:
```
conda update anaconda-navigator
conda build purge-all
```
to clear the conda source cache and run the installation again. 

### Install ffcv inside a currently existing environment
I do not want to create  a new environment for my current project, then I tried to install FFCV inside my current environment.
Look at [setup.py](https://github.com/libffcv/ffcv/blob/main/setup.py) and install required package.

I had to install these two packages separately.
```
conda install -c conda-forge libjpeg-turbo  # libturbojpeg
conda install -c conda-forge opencv  # cv4
```


### Converting your dataloader to FFCV dataloader
Looking from [main site](https://ffcv.io/), it looks easy to convert but you will notice many details were missed and you need to traverse back and forth over github repos ([main-repo](https://github.com/libffcv/ffcv), [ffcv--imagenet-repo](https://github.com/libffcv/ffcv-imagenet)) to know what to do.

I can say these important things are missing from this site (at 5:25 p.m Jul 21 2022 CT time).

- label metadata 
- how to convert dataset (e.g. ImageFolder) to beton format.


It may take time and cause daunting experience (the reason that I wrote this guide).

#### 1. Convert dataset to beton format
FFCV works with beton files to load the dataset.
I have been using `ImageFolder` to load my dataset. 

```
# Example
from torchvision import datasets
dataset = datasets.ImageFolder('/path/to/the/dataset/')
```

Normally, we pass the `image_transform` to `ImageFolder` to convert the PIL images to tensors but FFCV expects PIL objects, so don't pass transform options here.
Instead, we will pass it later in Loader() (i.e. to make the dataloader).
The code snippet blow converts Dataloader to beton files and save to your specified file name you specified.
```python
from torchvision import datasets
dataset = datasets.ImageFolder('/home/train/')

from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
writer = DatasetWriter(f'ffcv_output/imagenet_train.beton', {
    'image': RGBImageField(write_mode='jpg',
                           max_resolution=400,
                           compress_probability=0.5,
                           jpeg_quality=90),
    'label': IntField(),
},
    num_workers=16)
writer.from_indexed_dataset(dataset)  # This converts the dataset to beton files and saves
```

Here `write_mode` determines how to compress the images (`raw` - keep the same, `jpg` - JPEG compression, `smart` and `proportion` - partially compress. 

`max_resolution` determines how we resize the images. 

`compress_probability` is only used if `write_mode==proportion` and `jpeg_quality` determines the quality of jpeg encoding if we use `write_mode==jpg`.


__*Note: The size of the dataset can reduce/increase significantly if we change the options.

For example, using `write_mode=proportion` and `max_resolution=500` takes almost 10X times in memory in comparison with using `write_mode=jpg` and `max_resolution=400`. 
I think reducing the memory footprint (i.e. by using compression) is the key component why FCCV improve the training speed (and of course, potentially reduce the model performance).

#### 2. Create dataloader

Below is my dataloader.

```python
data_loader = Loader('ffcv_output/imagenet_{}.beton'.format(phase),
                     batch_size=RunningParams.batch_size,
                     num_workers=8,
                     order=OrderOption.RANDOM,
                     os_cache=True,
                     drop_last=True,
                     pipelines={'image': [
                      RandomResizedCropRGBImageDecoder((224, 224)),
                      RandomHorizontalFlip(),
                      ToTensor(),
                      ToDevice(torch.device('cuda:0'), non_blocking=True),
                      ToTorchImage(),
                      # Standard torchvision transforms still work!
                      NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, torch.float32)
                     ], 'label':
                     [
                        IntDecoder(),
                        ToTensor(),
                        Squeeze(),
                        ToDevice(torch.device('cuda:0'), non_blocking=True),
                            ]}
                     )
```

You can see it is differnt from the example from the [main website](https://ffcv.io/) as now I added the `'label` entry to `pipelines`.
You will also need to have some imports like below but I think they are mostly the wrappers of `torch` or `numpy` functions.

```python
from ffcv.loader import Loader, OrderOption
from ffcv.fields.decoders import \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from ffcv.transforms import ToTensor, Convert, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.transforms import *
import torchvision as tv
```

### Possible problems

#### 1. Half-tensors
If you get this ```HalfTensor``` error:

```
RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.cuda.FloatTensor) should be the same
```

please change the data type of your tensors to torch.float32 in Loader() function ([example]()).
The reason for this error is the ```torchvision``` pretrained models accept ```float32``` by default.
If you follow using torch.float16 like in the [ImageNet example](https://github.com/libffcv/ffcv-imagenet/blob/e97289fdacb4b049de8dfefefb250cc35abb6550/train_imagenet.py#L229), there will be a mismatch.

#### 2. Failed to import CuPy
When you run training, you may get problems with ```cupy```. Please locate your cuda version first by:

```
$ python
>>> import torch
>>> print(torch.version.cuda)
```

Mine was 11.3, then you can run ```pip install cupy-cuda113``` to get the corresponding ```cupy``` version. Please remember to remove any installed versions ```pip uninstall cupy-cudaxxx```. 

__*Note: Don't use ```nvidia-smi``` to get the cuda version because this command does NOT display the CUDA Toolkit version installed on your system (there could be multiple versions!).


#### 2. Slow training
TBD