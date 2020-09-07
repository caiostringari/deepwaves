# **Wave breaking statistics**


The file ```data.pkl``` contains the following information:

Variable | Description
---------|------------
`wave_breaking_area`     | Wave breaking area in squared meters.
`wave_breaking_duration`     | Wave breaking duration in seconds.
`wave_breaking_initial_speed`     | Initial wave breaking speed in meters per second.
`wave_breaking_mean_speed`     | Mean wave breaking speed in meters per second.
`wave_breaking_length`     | Wave breaking length in meters.
`wave_breaking_perimeter`     | Wave breaking length in meters (using the perimeter).
`ellipse_major_axis`     | Major ellipse axis (a) in meters (using `np.sum`)
`ellipse_minor_axis`     | Minor ellipse axis (b) in meters (using `np.sum`)
`max_ellipse_major_axis` | Major ellipse axis (a) in meters (using `np.max`). This is more reliable.
`max_ellipse_minor_axis` | Minor ellipse axis (a) in meters (using `np.max`). This is more reliable.
`wind_speed`    | Wind speed.
`reconstruction_area`    | Stereo video reconstruction area in squared meters.
`aquisition_length`    | Acquisition duration in seconds.
`peak_frequency_1`  | Peak wave frequency (first partition) in Hertz.
`peak_frequency_2`  | Peak wave frequency (second partition) in Hertz.

To reproduce the plot seen in the paper, do:

```bash
python make_plot.py
```

![](../docs/stats.png)
