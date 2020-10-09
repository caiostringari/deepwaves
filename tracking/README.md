# Wave Tracking

Two methods to track the waves in space time are currently implemented.

- Density Clustering
- [SORT](https://arxiv.org/abs/1602.00763)

SORT is orders of magnitude faster than clustering.

## Tracking via spatio-temporal clustering

To cluster wave breaking events in time and space use [```cluster.py```](cluster.py). This script can use the results of [```naive_wave_breaking_detector.py```](../util/naive_wave_breaking_detector.py) directly but this is not recommended. It is recommended that you narrow down the candidates for clustering using [```predict_from_naive_candidates.py```](../util/predict_from_naive_candidates.py) first.

**Example:**

```bash
python cluster.py -i "active_wave_breaking_events.csv" -o "clusters.csv" --cluster method "DBSCAN" --eps 10 -min-samples 10
```

<details>
  <summary> Arguments: </summary>

- ```-i [--input]``` Input path with images

- ```-o [--output]``` Output file name (see below for explanation).

- ```--cluster-method``` Either ```DBSCAN``` or ```OPTICS```. ```DBSCAN``` is recommended.

- ```--eps``` Mandatory parameter for ```DBSCAN``` or ```OPTICS```. See [here](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) for details.

- ```--min-samples``` Mandatory parameter for ```DBSCAN``` or ```OPTICS```. See [here](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) for details.

- ```--njobs``` Number of jobs to use.

- ```--chunk-size``` Maximum number of rows to process at a time. Default is 1000. Use lower values to avoid out-of-memory errors.

**Note**: The input data __*must*__ have at least the following entries: `ic`, `jc`, `ir`, `frame`.

</details>
</br>

The output of this script is a comma-separated value (csv) file. It looks like exactly like the output of [```naive wave breaking detector```](util/naive_wave_breaking_detector.py) with the addition of a column named ```wave_breaking_event```.

## Tracking using SORT

SORT can be used on the ellipses obtained with [```predict_from_naive_candidates.py```](../util/predict_from_naive_candidates.py) or on the detections created using image segmentation (see [```Image Segmentation```](../segmentation/README.md)).


**Example:**

```bash
python track.py -i "detections.csv" -o "tracks.csv"
```

<details>
  <summary> Arguments: </summary>

- ```-i [--input]``` Input detections file in csv format.

- ```-o [--output]``` Output file in csv format.

- ```--min-hits``` Minimum number of hits for SORT.

- ```--iou-threshold``` Intersection under union threshold.

- ```--max-age``` Maximum age to keep a track alive.

- ```--njobs``` Number of jobs to use.

- ```--from-ellipses``` Track from ellipses.

**Note**: The input data __*must*__ have at least the following entries: `ic`, `jc`, `ir`, `frame`.

</details>
</br>
