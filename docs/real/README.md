# Recorded data

Files in this folder are derived from SP-TransientBench (Dong et al., 2026, arXiv:2606.18952) and are licensed CC BY-NC-SA 4.0, as the dataset is. The rest of the repository is MIT. The raw data are not included: tools/fetch_stb.py downloads one sample to a folder outside version control.

- Dataset: <https://huggingface.co/datasets/Shao132/SP-TransientBench>
- Licence: <https://creativecommons.org/licenses/by-nc-sa/4.0/>

Cite the dataset as:

> Dong, H., Zhang, Z., Wen, Z., Qiang, Y., Deng, R., Dong, W., Jiang, Z., Li, X., Lu, R., Sun, S., Wang, W., Xia, Z., Zheng, H., Shi, G., & Ren, X. SP-TransientBench: A Real-Captured Single Photon Perception Benchmark. arXiv:2606.18952, 2026.

## What the files hold

All of them are computed from sample 1 of the DepthEstimate track by

```
python tools/fetch_stb.py --sample 1 --dest DIR
python tools/real_stb.py --data DIR --sample 1
```

The even columns of the sensor give the cycle count M, the response and the shift of the Livox projection, so `collapse.csv` and `shapes.png` come from them. `precision.csv`, `livox.csv` and `precision.png` are measured on the odd columns.

- `collapse.csv`: for every M tried, the mean L1 distance after Coates between the shapes of the 4 flux classes under 6400 photons, the ones the search for M uses. It is a diagnostic: the run takes M from the ceiling of the totals, the smallest multiple of 100 at or above their 99.9th percentile, because no pixel can record more photons than there were cycles, while where the shapes agree depends on the scene.
- `precision.csv`: for every flux class and estimator, `ratio` is the spread over 4 groups of each pixel against the first photon bound for one group, `bound_mm` the median of that bound as a distance, `groups` and `failures` the groups tried and lost, and `mean_photons` the mean recorded photons of the class. `pixels` counts the pixels that keep at least 2 of their groups for that estimator, so it differs between estimators: a pixel loses all its groups when an estimator refuses its whole histogram, as the matched filter does when the return lies within about 40 bins of the start of the record, the reach of its template. `ratio` and `bound_mm` are taken over those pixels.
- `livox.csv`: residuals against the Livox points on pixels with a compact reference and an estimate. Only the fullest bin uses the dataset's own conversion to range, so only its `median_cm` says something about the offset map. The other four estimators are each offset from that conversion, so their median is removed and reads 0. `spread_cm` is 1.4826 times the median absolute deviation from the median and `within5_pct` the share of pixels within 5 cm of it.
- `shapes.png`: the mean return of each of the 5 flux classes, as recorded and after Coates with M.
- `precision.png`: `ratio` against `mean_photons`, one line per estimator.
