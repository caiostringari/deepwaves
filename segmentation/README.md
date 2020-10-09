# Image Segmentation

Here, we use pre-trained and new models to identify active wave breaking
at the pixel level by using semantic segmentation neural nets. Two models are currently implemented, `Xception` and `MobileNetV2`. `Xception` is trained end-to-end and `MobileNetV2` is initialized from the weights of the image classification task.

Both models use a modified `U-Net` architecture. Data augmentation is used at training time.


## 1. Data

| Dataset | Link  | Alternative link  |
|-------|-------|-------------------|
**Segmentation V1** | [![](../badges/google_drive_badge.svg)](https://drive.google.com/file/d/1MPFswZoO_TVPewWFIxUD5yxapu-4mMzj/view?usp=sharing) | - |

The segmentation dataset was manually put together by me and Pedro Guimarães. Data is organized as follows:

```
└───train /or/ test /or/ valid
    ├───images
        ├───data
               ├───img1.png
               ├───img2.png
               ...
    ├───masks
        ├───data
               ├───img1.png
               ├───img2.png
               ...
```

## 2. Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yvL-egiE8Zj9S-4_zoV33QhrAj_wwnSE?usp=sharing) **|** [![Jupyter Notebook](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](../notebook/train_image_segmentation.ipynb)

### Xception

```bash
python train.py --backbone "xception" -i "path/to/data" --model "myxception" --logdir "myxception" --batch-size 32 --epochs 128
```

### MobileNetV2

```bash
python train.py --backbone "mobilenet" --pre-trained "mobilenet.h5" -i "path/to/data" --model "mymobilenet" --logdir "mymobilenet" --batch-size 32 --epochs 128
```

<details>
  <summary> Arguments: </summary>

  - `-i` Input data path.

  - `--model ` Model name.

  - `--backbone` Which backbone to use. Only mobilenet or xception for now.

  - `--random-state` Random seed for reproducibility. Default is 11.

  - `--epochs` Number of epochs (iterations) to train the model. Default is 200.

  - `--batch-size` Number of images to process in each step. Decrease if running into memory issues. Default is 64.

  - `--input-size` Image input size. Decrease if running into memory issues. Default is 256x256px.

  - `--pre-trained` Path to a pre-trained model.

</details>


## 3. Pre-trained models

| Model | Link  | Alternative link  |
|-------|-------|-------------------|
**Xception** | [![](../badges/google_drive_badge.svg)](https://drive.google.com/file/d/1Qko68JTZT-JLHKwSJJvvKUQEjmcy0V0j/view?usp=sharing) | - |
**MobileNetV2** | [![](../badges/google_drive_badge.svg)](https://drive.google.com/file/d/1uUcSW5s_jm5W-AQeeNxJKbIr6CR5fJIP/view?usp=sharing) | - |

## 4. Prediction on New Data

To segment new images, use `predict.py`. Note that the data will be processed in 256x256 blocks.

```bash
python predict.py --model "pre_trained/seg_xception.h5" --frames "path/to/images" -o "segmentation.csv" --region-of-interest 256 256 1024 512
```

<details>
  <summary> Arguments: </summary>

  - `--frames` Input data path.

  - `--model` Pre-trained model path.

  - `--region-of-interest` Region of the image to look at. Use Matplotlib convention.

  - `--regex` Regular expression used to look for frames.

  - `--from-frame` First frame.

  - `--frames-to-process` Number of frames to process.

  - `--output` Output csv file.

  - `save-plots` Will save output plots.

  - `plot-path` Where to save the plots.

</details>

## 5. Bounding Boxes Generation

It is also possible to obtain bounding boxes from the results of segmentation. This data can be used to track the waves using [SORT](https://arxiv.org/abs/1602.00763).

## Using scikit-learn `DBSCAN`

This is a slower approach but more precise.

```bash
python extract_detections_by_clustering.py --frames "path/to/frames" --pixels "segmentation.csv" -o "detections.csv" --plot --plot-path "plt" -eps 20 -nmin 10
```

<details>
  <summary> Arguments: </summary>

  - `--min-samples` minimum number of samples for DBSCAN.

  - `--eps` Maximum distance parameter for DBSCAN.

  - `--frames` Input data path.

  - `--pixels` Segmented pixels, use `predict.py` to get a valid file.

  - `--regex` Regular expression used to look for frames.

  - `--from-frame` First frame.

  - `--frames-to-process` Number of frames to process.

  - `--output` Output csv file.

  - `save-plots` Will save output plots.

  - `plot-path` Where to save the plots.

</details>


## Using scikit-image `label` and `regionprops`

This is faster but a lot less precise.

```bash
python extract_detections_by_labelling.py --frames "path/to/frames" --pixels "segmentation.csv" -o "detections.csv" --plot --plot-path "plt" -min-area 20
```

<details>
  <summary> Arguments: </summary>

  - `--min-area` minimum number area to consider as a region.

  - `--frames` Input data path.

  - `--pixels` Segmented pixels, use `predict.py` to get a valid file.

  - `--regex` Regular expression used to look for frames.

  - `--from-frame` First frame.

  - `--frames-to-process` Number of frames to process.

  - `--output` Output csv file.

  - `save-plots` Will save output plots.

  - `plot-path` Where to save the plots.

</details>
