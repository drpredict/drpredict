# A Deep Learning System for Detecting Diabetic Retinopathy across the Disease Spectrum

## Contents

1. Requirements
2. Environment setup
    * Linux System
    * Docker
3. Preparing the dataset
4. Train and Testing

## Requirements

This software requires a **Linux** system: [**Ubuntu 20.04**](https://ubuntu.com/download/desktop) or  [**Ubuntu 18.04**](https://ubuntu.com/download/desktop) (other versions are not tested)   and  [**Python3.8**](https://www.python.org) (other versions are not supported). This software requires **8GB memory** and **10GB disk** storage (we recommend 16GB memory). The software analyzes a single image in **5 seconds** on **Intel(R) Core(TM) i7-8700K CPU @ 3.70GHz**. Some **linux packages** are required to run this software as listed below:

```
build-essential
zlib1g-dev
libncurses5-dev
libgdbm-dev
libnss3-dev
libssl-dev
libreadline-dev
libffi-dev
libsqlite3-dev
libsm6
libxrender1
wget
git
```

The **python packages** needed are listed below. They can also be found in `reqirements.txt`.

```
Click         >=7.0
numpy         >=1.18.1
opencv-python >=4.2.0.32
Pillow        >=6.1.0
setuptools    >=41.2.0
six           >=1.14.0
torch         >=1.4.0
torchvision   >=0.5.0
```

## Native Environment Setup

### Linux System

#### Step 1: download the project
1. Open the terminal in the system, or press Ctrl+Alt+F1 to switch to the command line window.
1. Clone this repo file to the home path.

```
git clone https://github.com/drpredict/predict.git
```

3. change the current directory to the source directory

```
cd predict
```

#### Step 2: prepare the running environment and run the code***

1. install dependent Python packages

```
python3 -m pip install --user -r requirements.txt
```
#### Notice: IF THE PYTHON IS NOT PYTHON 3.8, INSTALL PYTHON 3.8 FIRST


### Docker Environment Setup

It is recommended to use docker to run this software. We provided the docker image that contains all dependency and requirements. Please follow [this link](https://docs.docker.com/docker-for-windows/install/) to install Docker Engine CE, and follow [this link](https://docs.docker.com/compose/install/) to install Docker Compose.

The Dockerfile is also provided in this repository. All dependent software and packages are installed and set up in the docker image.

**Supported Image File Format**

JPG, PNG, and TIF formats are supported and tested. Other formats that OpenCV can decode should work but not tested. The input image must be 3-channel color fundus images with a small edge resolution larger than 448.


## preparing the dataset

For classification tasks, the data folder structure is similar to the image to ImageNet. There are subdirectories in the dataset root directory, and each subdirectory represents a class. All images of a specific class are put into the subdirectory representing the class.

For segmentation tasks, all fundus images and mask labels are put in the same folder. The name of the fundus image should be xxxx.jpg. And for each lesion (microaneurysm, hard exudates, soft exudates, hemorrhages), the file name should be xxxxx_MA.tif, xxxxx_EX.tif, xxxxx_SE.tif, and xxxxx_HE.tif separately. Where xxxxx is the sample index (e.g. 00001)

## Train and Testing
For easy use of the code, we provided a simple command-line tool in `main.py`.

All options are listed in the help documents. Refer to `main.py` for more details. The following instructions can be used to train models:

1. The following command can be used to to train the BaseDRModule on your data: `pythonmain.py --train_classification --data_root where_your_data_is_stored --lr 0.001 --dump where_you_put_the_classification_result_model`
1. The following command can be used to to train the MaskRCNN on your data: `python main.py --train_maskrcnn --data_root where_your_data_is_stored --lr 0.0001 --dump where_you_want_to_put_the_result_model`
1. The `--load_and_transfer where_you_stored_the_classification_weight` option will load the pre-trained network into the maskcnn as initial weight.
1. The `--dump where_to_store_dump_file` will specify the path to store the dump file, three files will be created for the overall weights, classification weights, and maskrcnn weights separately.
1. The `--load_all`, `--load_maskrcnn`, `--load_classification` will load the trained weights to the specified module.
1. The `--lr` option will set the base learning rate.
1. The `--with_maskrcnn` option is used when training the classification network. If `--with_maskrcnn` is specified, then the maskcnn feature will be fused with classification.
1. The `--max_epoch` option specifies the number of epoch to train.
1. The '--trainable_layers' option sets the number of trainable (not frozen) res-net layers starting from the final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.

To train the whole system, you need to:
1. Train the BaseDR model with `python main.py --train_classification  --data_root dr_grade_training_data --lr 0.001 --dump base_dr`
2. Train the quality model with `python main.py --train_classification  --data_root quality_data --lr 0.001 --dump quality --load_classification base_dr_classification.pkl`
3. Train the lesion detection with `python main.py --train_classification  --data_root lesion_detection_training_data --lr 0.001 --dump lesion_detection --load_classification base_dr_classification.pkl`
4. Train the lesion segmentation with `python main.py --train_maskrcnn --data_root lesion_segmentation_dsata --lr 0.0003 --dump lesion_segmentation --load_transfer base_dr_classification.pkl`
5. Train the final DR grading network with `python main.py --train_classification --with_maskrcnn --data_root dr_grade_training_data --lr 0.001 --dump dr_model --load_classification base_dr_classification.pkl --load_maskrcnn lesion_segmentation_mask.pkl --trainable_layers 0 `
