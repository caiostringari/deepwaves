# Deep Neural Networks for Active Wave Breaking Classification

This repository contains code and data to reproduce the results of the paper **Deep Neural Networks for Active Wave Breaking Classification** currently under review.


[![](badges/overleaf_badge.svg)](https://www.overleaf.com/read/mhprcfwhryfw) **|** [![](badges/arxiv_badge.svg)](https://www.overleaf.com/read/mhprcfwhryfw)

## Dependencies

```bash
# create a new environment
conda create --name tf python=3.7

# activate your new environment
conda activate tf

# If you have a nvidia GPU installed and properly configured
conda install tensorflow-gpu=2
conda install tensorboard

# Natsort - better file sorting
conda install natsort

# Classical machine learning
conda install pandas scikit-learn scikit-image

# Extra thresholding methods
pip install pythreshold

# fitting circles to data
pip install miniball

# parallel computations
pip install pebble

# Matplotlib and seaborn
conda install matplotlib seaborn

# netCDF support
conda install netCDF4 xarray

# make your life easier with ipython
conda install ipython
```

- [Deep Neural Networks for Active Wave Breaking Classification](#deep-neural-networks-for-active-wave-breaking-classification)
  * [Dependencies](#dependencies)
  * [1. Data](#1-data)
    + [1.1. Published data](#11-published-data)
    + [1.2. Creating a dataset from the scratch](#12-creating-a-dataset-from-the-scratch)
      - [1.2.1. Extracting wave breaking candidates](#121-extracting-wave-breaking-candidates)
      - [1.2.2. Sampling wave breaking candidates for training](#122-sampling-wave-breaking-candidates-for-training)
      - [1.2.3. Merging Datasets](#123-merging-datasets)
  * [2. Models](#2-models)
      - [2.1. Training the Neural Network](#21-training-the-neural-network)
      - [2.2. Pre-trained Models](#22-pre-trained-models)
  * [3. Evaluating Model Performance](#3-evaluating-model-performance)
  * [4. Using a Pre-trained Neural Network](#4-using-a-pre-trained-neural-network)
  * [5. Results](#5-results)
  * [6. Appendix I (Standard Variable Names)](#6-standard-variable-names)
  * [Disclaimer](#disclaimer)

## 1. Data

### Manually

- Refer to [Manual Data Preparation](util/README.md).

### Production ready


| Model | Link  | Alternative link  |
|-------|-------|-------------------|
**Train (10k)** | [![](badges/google_drive_badge.svg)](https://drive.google.com/file/d/1Qko68JTZT-JLHKwSJJvvKUQEjmcy0V0j/view?usp=sharing) | - |
**Train (20k)** | [![](badges/google_drive_badge.svg)](https://drive.google.com/file/d/1uUcSW5s_jm5W-AQeeNxJKbIr6CR5fJIP/view?usp=sharing) | - |
**Test (1k)** | [![](badges/google_drive_badge.svg)](https://drive.google.com/file/d/1A6IK9IQjFN9JMNx3bUkcWdlO8YN8PbaC/view?usp=sharing) | - |
**Black Sea (200k)** | [![](badges/google_drive_badge.svg)](?) | - |
**La Jument 2019 (100k)** | [![](badges/google_drive_badge.svg)]() | - |

**Note:** The models described in the paper and in this documentation were trained using the 20k Dataset.

## Standard Variable Names

The following variables are standard across this repository and scripts that output these quantities should use these names. If a given script has extra output variables, these are documented in each script.

| Variable              | Description                                                                                                                                                  |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `x`                   |  x-coordinate in metric coordinates.                                                                                                                         |
| `y`                   |  y-coordinate in metric coordinates.                                                                                                                         |
| `z`                   |  z-coordinate in metric coordinates.                                                                                                                         |
| `time`                |  date and time. Use a format that [pandas.to_datetime()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html) can understand. |
| `frame`               |  sequential number.                                                                                                                                          |
| `i`                   |  pixel coordinate in pixel units. Use [Matplotlib coordinate system](https://matplotlib.org/3.1.1/tutorials/intermediate/imshow_extent.html).                |
| `j`                   |  pixel coordinate in pixel units. Use [Matplotlib coordinate system](https://matplotlib.org/3.1.1/tutorials/intermediate/imshow_extent.html).                |
| `ic`                  |  center of a circle or ellipse in pixel coordinates.                                                                                                         |
| `jc`                  |  center of a circle or ellipse in pixel coordinates.                                                                                                         |
| `xc`                  |  center of a circle or ellipse in metric coordinates.                                                                                                        |
| `yc`                  |  center of a circle or ellipse in metric coordinates.                                                                                                        |
| `ir`                  |  radius in the i-direction.                                                                                                                                  |
| `jr`                  |  radius in the j-direction.                                                                                                                                  |
| `xr`                  |  radius in the x-direction.                                                                                                                                  |
| `yr`                  |  radius in the y-direction.                                                                                                                                  |
| `theta_ij`            |  angle of rotation of an ellipse with respect to the x-axis counter-clockwise.                                                                               |
| `theta_xy`            |  angle of rotation of an ellipse with respect to the x-axis counter-clockwise.                                                                               |
| `wave_breaking_event` |  unique wave breaking event id.                                                                                                                              |
| `vx`                  |  velocity in the x-direction in m/s.                                                                                                                         |
| `vy`                  |  velocity in the y-direction in m/s.                                                                                                                         |
| `vi`                  |  velocity in the x-direction in pixels/frame.                                                                                                                |
| `vj`                  |  velocity in the y-direction in pixels/frame.                                                                                                                |                                 |

## Disclaimer

There is no warranty for the program, to the extent permitted by applicable law except when otherwise stated in writing the copyright holders and/or other parties provide the program “as is” without warranty of any kind, either expressed or implied, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose. the entire risk as to the quality and performance of the program is with you. should the program prove defective, you assume the cost of all necessary servicing, repair or correction.



<!--

### 1.1. Published data

Use the following links to download the pre-defined datasets.

- **Train** | [Google Drive](https://drive.google.com/open?id=1xa_cdgc0MYD_e_OFZyo0ppzMnXLffQEB) | [Zenodo](Upcoming)
- **Test** | [Google Drive](https://drive.google.com/open?id=1OJ7nWrYeqekEiluwltU0mKmcybSFvO0b) | [Zenodo](Upcoming)

### 1.2. Creating a dataset from the scratch

#### 1.2.1. Extracting wave breaking candidates

If you wish to start creating a dataset from the scratch, first you need to obtain
wave breaking candidates. For this task use the
[naive wave breaking detector](src/naive_wave_breaking_detector.py). It will naively detect wave breaking using an adaptive thresholding approach.

For help: ```python naive_wave_breaking_detector.py --help```

**Example:**

```bash
python naive_wave_breaking_detector.py --debug --input "input/folder/" --output "output" --subtract-averages "average/folder" --cluster "dbscan" 10 10 --threshold "adaptive" 11 10 --region-of-interest "file.csv" --temporary-path "tmp/" --fit-method "ellipse" --nproc 4
```

**Options:**

- ```--debug```: Runs in debug mode and will save output plots.

- ```-i [--input]```: Input path with images

- ```-o [--output]```: Output file name (see below for explanation).

- ```--subtract-averages```: Input path with pre-computed average images. Use [compute average image](src/compute_averaged_image.py) to generate valid files.

- ```--cluster``` Clustering method. Only [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) is fully functional. Must also inform `eps` and `n_min` parameters.

- ```--threshold```: Thresholding method. Valid options are: `file`, `constant`, `entropy`, `otsu` and `adaptative`. Default is `adaptative` with window size 11 and offset 10.

- ```--region-of-interest```: File with region of interest. Use [Minimun Bounding Geometry](src/minimum_bounding_geometry.py) to generate a valid input file.

- ```--temporary-path```: Path to write temporary files. Will save plots to this path if in debug mode.

- ```--fit-method``` Which geometry to fit to a detected cluster of bright pixels in the image. Valid options are *circle* or *ellipse*. If an ellipse cannot be fitted to the data, will fall back to fitting a circle.

- ```--nproc```: Number of processors to use. If ```debug``` is parsed, defaults to one.

- ```--block-shape 1024 1024```: Block shape to split the image into to avoid memory errors.

- ```--frames-to-plot```: Number of frames to plot if in debug mode.

- ```-timeout```: If in parallel mode, kill a processes if its taking longer than 120 seconds per default. This helps to avoid out-of-memory issues caused by DBSCAN.

- ```--force-plot-in-parallel-mode```: Will plot outputs even if in parallel mode. Useful to debug even faster.

**Output:**

The output is this script is a comma-separated value (csv) file. See below for variable name explanation. The extra variable names included in the output file are:


| Variable             | Description                                           |
|----------------------|-------------------------------------------------------|
| ```pixels```         |  Number of pixels in each cluster                     |
|  ```cluster```       |  Cluster label from either `DBSCAN` or `OPTICS`       |
| ```block_i```        |   Block index in the `i`-direction                    |
| ```block_j```        |  Block index in the `j`-direction                     |
| ```block_i_left```   |  Block start in the `i`-direction (image referential) |
| ```block_i_right```  |  Block end  in  the `i`-direction (image referential) |
| ```block_j_top```    |  Block end  in  the `j`-direction (image referential) |
| ```block_j_bottom``` |  Block start the `j`-direction (image referential)    |


Graphically, the results of this script looks like this:

![](docs/naive_detector.gif)

#### 1.2.2. Sampling wave breaking candidates for training

Once you have some wave breaking candidates, use `prepare_data_for_classifer.py`
to extract random samples:

**Example:**

```bash
python prepare_data_for_classifer.py -i "wave_breaking_candidates.csv" --frames "path/to/frames" -o "path/to/output" -N 100 -size 256 256 --region-of-interest "region_of_interest.csv"
```

**Options:**

- `-i [--input]` Input data obtained with `naive_wave_breaking_detector` or
other feature extraction program.

- `--frames [-frames]` Input path with extracted frames.

- `--region-of-interest` Region of interest file.

- `--samples [-n, -N]` Number of random samples to extract.

- `-o [--output]` Output path.

- `--size [-size]`  Window size and output image size.

The `output` folder has the following structure:

```
output
  ├───img
  |───plt
  |───labels.csv
```

- `img` Contains the image data that is used as input data for the neural network.

- `plt` Contains plots of the extract samples.

- `labels.csv` This file has the same structure as the results from [naive wave breaking detector](src/naive_wave_breaking_detector.py) but with the addition of a column

- `class` the user has to manually fill assigning the correct labels by looking at the plots.


#### 1.2.3. Merging Datasets

To merge multiple datasets created with `prepare_data_for_classifer.py` you can use
'merge_data_for_classifier.py'.

This script can handle multi-label inputs but the out is always binary. Use the `target-labels` option to tell the script which labels should be considered as `True` (`1`) in the output.

**Example:**

```bash
python merge_data_for_classifier.py -i "path/to/dataset1/" "path/to/dataset2" -o "path/to/output/" --target-labels 4 5
```

***Warning***: There is possible bug in this script. Fixing for the next version.

**Arguments:**

- `-i [--input]` Input data paths created with `prepare_data_for_classifer.py`

- `-o [--output]` Output path.

- `target-labels` Which labels to consider `True` in the binarization process.

Optional:

- `-crop-size` Will crop input images to a desired size if asked.


## 2. Models

#### 2.1. Training the Neural Network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1b7h90t3EJx91UTyzCQq8YSyTYzW_lJnZ?usp=sharing) **|** [![Jupyter Notebook](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](notebook/train_wave_breaking_classifier_v2.ipynb)

**Note**: The training dataset used here is a smaller version (10k) of the published dataset.

This program loads manually labeled wave image data and classify
the images into "breaking" (1) or "no-breaking" (0).

The data needs to be in a folder which has sub-folders "0" and "1"

For example:
```
train
    ├───0
    ├───1
```

There are 5 `backbones` implemented: `VGG16`, `ResNet50V2`, `InceptionResNetV2`, `MobileNetV2` and `EfficientNet`

Note that the weights from these pre-trained models will be reset and
updated from the scratch here.

These models have no knowledge of the present data and, consequently,
transfered learning does not work well.

**Example**

```bash
python train_wave_breaking_classifier_v2.py --data "train/" --backbone "VGG16" --model "vgg_test" --logdir "logs/" --random-state 11 --validation-size 0.2 --learning-rate 0.00001 --epochs 200 --batch-size 200 --dropout 0.5 --input-size 256 256
```

**Arguments:**

- `--data` Input train data path.

- `--model ` Model name.

- `--backbone` Which backbone to use. See above.

Optional:

- `--random-state` Random seed for reproducibility. Default is 11.

- `--validation-size` Size of the validation dataset. Default is 0.2.

- `--epochs` Number of epochs (iterations) to train the model. Default is 200.

- `--batch-size` Number of images to process in each step. Decrease if running into memory issues. Default is 64.

- `--dropout` Droput percentage. Default is 0.5.

- `--input-size` Image input size. Decrease if running into memory issues. Default is 256 256.


The neural network looks something like this:

![](docs/cnn.png)

#### 2.2. Pre-trained Models

Please use the links below to download pre-trained models:

- [**VGG16**]()
- [**ResNet50V2**]()
- [**InceptionResNetV2**]()
- [**MobileNet**](https://drive.google.com/open?id=1N0N03QDevACbOAi0Vq9tAMShYBX6s8EA) | [**Training History**](https://drive.google.com/open?id=194R0zOyNvh7z5AZqRnxHS34_vEQ_r6un)
- [**EfficientNet**]()


## 3. Evaluating Model Performance

To evaluate a pre-trained model on test data, use the [test wave breaking classifier](src/test_wave_breaking_classifier.py) script.

**Example:**

```bash
python test_wave_breaking_classifier.py --data "path/to/test/data/" --model "VGG16.h5" --threshold 0.5 -- output "path/to/results.csv"
```

**Arguments:**

- `--data` Input test data path created with [merge data for classifier](src/merge_data_for_classifier.py) or manually.

- `--model ` Pre-trained model.

- `--threshold` Threshold for binary classification. Default is 0.5

- `--output` path to save the results.

The `classification report` with be printed on the screen. For example:

```
              precision    recall  f1-score   support

         0.0       0.88      0.99      0.94      1025
         1.0       0.87      0.23      0.36       175

    accuracy                           0.88      1200
   macro avg       0.88      0.61      0.65      1200
weighted avg       0.88      0.88      0.85      1200
```

To summarize the model metrics do:

```bash
python sumarize_model_metrics.py --data "path/to/data/" --model "VGG16.h5" --threshold 0.5 -- output "path/to/metrics.csv"
```

The arguments are the same as above.

The results look something like this:

|VGG16    |Binary_Accuracy|True_Positives|False_Positives|True_Negatives|False_Negatives|Precision|Recall|AUC |
|----------|---------------|--------------|---------------|--------------|---------------|---------|------|----|
|Train     |0.89           |771.00        |248.00         |5680.00       |521.00         |0.76     |0.60  |0.92|
|Validation|0.87           |100.00        |19.00          |1463.00       |222.00         |0.84     |0.31  |0.90|
|Test      |0.88           |40.00         |6.00           |1019.00       |135.00         |0.87     |0.23  |0.82|

To plot the training curves and a confusion matrix, do:

```bash
python plot_history_and_confusion_matrix.py --history "path/to/history.csv" --results "path/to/results.csv" --output "figure.png"
```

**Arguments:**

- `--history` Training history. Comes from `train_wave_breaking_classifier_v2.py`.

- `--results ` Classification results from the test data. Comes from `test_wave_breaking_classifier.py`.

- `--output` Figure name.

The results look like this:
![](docs/hist_cm.png)


## 4. Using a Pre-trained Neural Network

### 4.1 Using new data

Create a dataset either manually or with the provided tools. The data structure is as follows:

```
pred
    ├───images
        ├───img_00001.png
        ├───img_00002.png
        ├───...
        ├───img_0000X.png
```

```bash
python predict.py --data "pred/" --model "VGG16.h5" --threshold 0.5 --output "results.csv"
```

**Arguments:**

- `--data` Input test data.

- `--model ` Pre-trained model.

- `--threshold` Threshold for binary classification. Default is 0.5

- `--output` A csv file with the classification results.

### 4.2. Using the results from the naïve detector

Use the results from the [naive wave breaking detector](src/naive_wave_breaking_detector.py) and a pre-trained neural network
to obtain only **active wave breaking** instances. This script runs on ```CPU``` but can be much faster on ```GPU```.

For help: ```python predict_from_naive_candidates.py --help```

**Example**

```bash
python predict_active_wave_breaking_v2.py --debug --input "naive_results.csv" --model "path/to/model.h5" --frames "path/to/frames/folder/"  --region-of-interest "region_of_interest.csv" --output "robust_results.csv" --temporary-path "tmp" --frames-to-plot 1000 --threshold 0.5
```

**Arguments**

- ```--debug``` Runs in debug mode and will save output plots.

- ```-i [--input]``` Input data obtained from ```naive_wave_breaking_detector```.

- ```-m [--model]``` Pre-trained Tensorflow model. See [here](https://drive.google.com/open?id=1b7h90t3EJx91UTyzCQq8YSyTYzW_lJnZ).

- ```-o [--output]``` Output file name (see below for explanation).

- ```-frames [--frames]``` Input path with images.

- ```--region-of-interest``` File with region of interest. Use [minimun bounding geometry](src/minimum_bounding_geometry.py) to generate a valid input file.

- ```-temporary-path``` Output path for debug plots.

- ```--frames-to-plot``` Number of frames to plot.

- ```--threshold``` Threshold for activation in the last (sigmoid) layer of the model. Default is `0.5`.

***Note:*** The input data __*must*__ have at least the following entries: `ic`, `jc`, `ir`, and `frame`.


**Output:**

The output of this script is a comma-separated value (csv) file. It looks like exactly like the output of [naive wave breaking detector](src/naive_wave_breaking_detector.py) but only with data considered as **active wave breaking**.

Graphically, the results of this script looks like this:

![](docs/robust_detector.gif)

## 5. Results

The table below summarizes the results presented in the paper. Results are sorted by AUC.

**Train**


| Model             | Accuracy | TP   | FP  | TN    | FN   | Precision | Recall | AUC   |
|-------------------|----------|------|-----|-------|------|-----------|--------|-------|
| ResNetV250        | 0.97  | 1414 | 198  | 13978 | 280  | 0.877 | 0.835 | 0.989 |
| VGG16             | 0.93  | 855  | 273  | 13911 | 831  | 0.758 | 0.507 | 0.943 |
| InceptionResnetV2 | 0.927 | 886  | 359  | 13823 | 802  | 0.712 | 0.525 | 0.932 |
| EfficientNet      | 0.772 | 1403 | 3346 | 10920 | 297  | 0.295 | 0.825 | 0.874 |
| MobileNet         | 0.904 | 436  | 268  | 13916 | 1250 | 0.619 | 0.259 | 0.848 |


**Validation**

| Model             | Accuracy | TP   | FP  | TN    | FN   | Precision | Recall | AUC   |
|-------------------|----------|------|-----|-------|------|-----------|--------|-------|
| VGG16             | 0.932 | 221 | 65  | 3478 | 204 | 0.773 | 0.52  | 0.946 |
| ResNetV250        | 0.919 | 197 | 97  | 3450 | 224 | 0.67  | 0.468 | 0.873 |
| InceptionResnetV2 | 0.921 | 190 | 81  | 3466 | 231 | 0.701 | 0.451 | 0.93  |
| EfficientNet      | 0.809 | 353 | 687 | 2856 | 72  | 0.339 | 0.831 | 0.897 |
| MobileNet         | 0.908 | 123 | 64  | 3479 | 302 | 0.658 | 0.289 | 0.878 |


**Test**

| Model             | Accuracy | TP   | FP  | TN    | FN   | Precision | Recall | AUC   |
|-------------------|----------|------|-----|-------|------|-----------|--------|-------|
| VGG16             | 0.876 | 106 | 80 | 945  | 69  | 0.57  | 0.606 | 0.855 |
| ResNetV250        | 0.881 | 95  | 63 | 962  | 80  | 0.601 | 0.543 | 0.843 |
| InceptionResnetV2 | 0.882 | 91  | 57 | 968  | 84  | 0.615 | 0.52  | 0.839 |
| EfficientNet      | 0.873 | 88  | 65 | 960  | 87  | 0.575 | 0.503 | 0.827 |
| MobileNet         | 0.875 | 30  | 5  | 1020 | 145 | 0.857 | 0.171 | 0.768 |


## 6. Standard Variable Names

The following variables are standard and all scripts that output these quantities should use these names.

| Variable              | Description                                                                                                                                                  |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `x`                   |  x-coordinate in metric coordinates.                                                                                                                         |
| `y`                   |  y-coordinate in metric coordinates.                                                                                                                         |
| `z`                   |  z-coordinate in metric coordinates.                                                                                                                         |
| `time`                |  date and time. Use a format that [pandas.to_datetime()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html) can understand. |
| `frame`               |  Sequential number.                                                                                                                                          |
| `i`                   |  pixel coordinate in pixel units. Use [Matplotlib coordinate system](https://matplotlib.org/3.1.1/tutorials/intermediate/imshow_extent.html).                |
| `j`                   |  pixel coordinate in pixel units. Use [Matplotlib coordinate system](https://matplotlib.org/3.1.1/tutorials/intermediate/imshow_extent.html).                |
| `ic`                  |  center of a circle or ellipse in pixel coordinates.                                                                                                         |
| `jc`                  |  center of a circle or ellipse in pixel coordinates.                                                                                                         |
| `xc`                  |  center of a circle or ellipse in metric coordinates.                                                                                                        |
| `yc`                  |  center of a circle or ellipse in metric coordinates.                                                                                                        |
| `ir`                  |  radius in the i-direction.                                                                                                                                  |
| `jr`                  |  radius in the j-direction.                                                                                                                                  |
| `xr`                  |  radius in the x-direction?                                                                                                                                  |
| `yr`                  |  radius in the y-direction?                                                                                                                                  |
| `theta_ij`            |  Angle of rotation of an ellipse with respect to the x-axis counter-clockwise.                                                                               |
| `theta_xy`            |  Angle of rotation of an ellipse with respect to the x-axis counter-clockwise.                                                                               |
| `wave_breaking_event` |  unique wave breaking event id.                                                                                                                              |
| `vx`                  |  Velocity in the x-direction in m/s.                                                                                                                         |
| `vy`                  |  Velocity in the y-direction in m/s.                                                                                                                         |
| `vi`                  |  Velocity in the x-direction in pixels/frame.                                                                                                                |
| `vj`                  |  Velocity in the y-direction in pixels/frame.                                                                                                                |                                 |

-->
