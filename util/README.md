# **Tools**

- [1. **Wave Breaking Detection: data preparation**](#1---wave-breaking-detection--data-preparation--)
  * [1.1. Extracting data](#11-extracting-data)
  * [1.2. Sampling wave breaking candidates for training](#12-sampling-wave-breaking-candidates-for-training)
  * [1.3. Merging Datasets](#13-merging-datasets)

## 1. **Wave Breaking Detection: data preparation**

### 1.1. Extracting wave breaking candidates from raw data

This is the main wave breaking detection script. It will naively detect wave breaking
using an default adaptive thresholding approach. This script can also be used to generate binary
masks that can be used by other algorithms.

For help: ```python naive_wave_breaking_detector.py --help```

**Example:**

```bash
python naive_wave_breaking_detector.py --debug --input "input/folder/" --output "output" --subtract-averages "average/folder" --cluster "dbscan" 10 10 --threshold "file" "file.csv" --region-of-interest "file.csv" --temporary-path "tmp" --fit-method "ellipse" --nproc 4
```

**Arguments:**

- ```--debug```: Runs in debug mode and will save output plots.

- ```-i [--input]```: Input path with images

- ```-o [--output]```: Output file name (see below for explanation).

- ```--subtract-averages```: Input path with pre-computed average images. Use [compute average image](compute_averaged_image.py) to generate valid files.

- ```--cluster``` Clustering method. Only [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) is fully tested. Must also inform `eps` and `n_min` parameters.

- ```--threshold```: Thresholding method. Valid options are: `file`, `constant`, `entropy`, `otsu` and `adaptative`. Default is `adaptative` with window size 11 and offset 10.

- ```--region-of-interest```: File with region of interest. Use [Minimun Bounding Geometry](minimum_bounding_geometry.py) to generate a valid input file.

- ```--temporary-path```: Path to write temporary files. Will save plots to this path if in debug mode.

- ```--fit-method``` Which geometry to fit to a detected cluster of bright pixels in the image. Valid options are *circle* or *ellipse*. If an ellipse cannot be fitted to the data, will fall back to fitting a circle.

- ```--nproc```: Number of processors to use. If ```debug``` is parsed, defaults to one.

- ```--block-shape 1024 1024```: Block shape to split the image into to avoid memory errors.

- ```--frames-to-plot```: Number of frames to plot if in debug mode.

- ```-timeout```: If in parallel mode, kill a processes if its taking longer than 120 seconds per default. This helps to avoid out-of-memory issues caused by DBSCAN.

- ```--force-plot-in-parallel-mode```: Will plot outputs even if in parallel mode. Useful to debug even faster.

**Output:**

The output is this script is a comma-separated value (csv) file. See above for variable name explanation. The extra variable names included in the output file are:


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

![](../../doc/naive_detector.gif)

### 1.2. Sampling wave breaking candidates for training

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

- `labels.csv` This file has the same structure as the results from [naive wave breaking detector](../../stereo_video/breaking_detection/naive_wave_breaking_detector.py) but with the addition of a column

- `class` the user has to manually fill assigning the correct labels by looking at the plots.


### 1.3. Merging Datasets

To merge multiple datasets created with `prepare_data_for_classifer.py` you can use
'merge_data_for_classifier.py'.

This script can handle multi-label inputs but the out is always binary. Use the `target-labels` option to tell the script which labels should be considered as `True` (`1`) in the output.

**Example:**

```bash
python merge_data_for_classifier.py -i "path/to/dataset1/" "path/to/dataset2" -o "path/to/output/" --target-labels 4 5
```

***Warning***: There is possible bug in this script. Trying to fix for a next version.

**Arguments:**

- `-i [--input]` Input data paths created with `prepare_data_for_classifer.py`

- `-o [--output]` Output path.

- `target-labels` Which labels to consider `True` in the binarization process.

Optional:

- `-crop-size` Will crop input images to a desired size if asked.
