# Camera-to-body extrinsic refinement — KITTI POC

Proof of concept for **targetless camera calibration against a known lidar**. The lidar-to-body
transform is treated as known; the camera-to-body transform is refined by aligning lidar edges with
image edges. Everything is in a single script, `calib_poc.py`.

KITTI raw stands in for the general vehicle:

| general vehicle            | KITTI raw                              |
|-------------------------|----------------------------------------|
| vehicle body `B`        | OXTS IMU frame                         |
| lidar `L`, `T_B_L` known | Velodyne HDL-64, `calib_imu_to_velo.txt` |
| camera `C`, `T_C_B` unknown | cam0 (unrectified) from `calib_velo_to_cam.txt`; projected with `P_rect_02 · R_rect_00` |

The ground-truth `T_C_B = T_C_L · T_L_B` comes from the KITTI calibration files. The benchmark
perturbs it, runs the optimizer, and measures how close the result gets to ground truth.

## Running

Requires [`uv`](https://docs.astral.sh/uv/). Dependencies (numpy, scipy, opencv) are declared
inline in the script, so no environment setup is needed.

```sh
# 1. Download calibration + one drive (~650 MB for drive 0005)
uv run calib_poc.py download --root ~/data/kitti

# 2. Single runs with overlays: perturb GT, recover it, write init/estimate/GT overlay images
uv run calib_poc.py run --root ~/data/kitti --trials 5 --rot-deg 2 --trans-m 0.03

# 3. Benchmark sweep: every (rotation error x translation error x trial), every stage
uv run calib_poc.py bench --root ~/data/kitti --trials 20
uv run calib_poc.py bench --root ~/data/kitti --trials 20 --rot-list 1,2,4 --opt-trans
```

| Option | Default | Meaning |
|---|---|---|
| `--root` | `~/data/kitti` | KITTI raw root directory |
| `--date`, `--drive` | `2011_09_26`, `0005` | Recording; `--drive` accepts a comma-separated list (same date) |
| `--cam` | `2` | KITTI camera index (2 = left colour) |
| `--frames` | `10` | Frames per drive, evenly spaced |
| `--max-speed` | `inf` | Use only frames at or below this vehicle speed (m/s, from OXTS) |
| `--rot-deg`, `--trans-m` | `2.0`, `0.03` | `run`: injected rotation (°) / translation (m) error, random direction |
| `--rot-list`, `--trans-list` | `0.5,1,2,3,4,6`, `0,0.03,0.1` | `bench`: error sweep |
| `--trials` | `3` | Random trials per setting |
| `--opt-trans` | off | Also optimize translation (see results: weakly observable) |
| `--no-p2l` | off | `run`: report the stage B result instead of stage C |
| `--success-deg` | `0.5` | `bench`: success threshold on rotation error |
| `--out` | `calib_poc_out` | Output directory (overlays, `bench.csv`) |

**Outputs**
- `run`: a per-trial table (rotation error °, translation error m, median edge distance px) and
  `trial<N>.png` overlays stacked as *init / estimate / GT*. Red = image edges, green = projected lidar edges.
- `bench`: a summary table per (error, stage) plus `bench.csv` with one row per trial and stage:
  `rot_in, trans_in, trial, stage, rot_err, pitch, yaw, roll, trans_err, px, secs`.

## Algorithm

The unknown is `T_C_B`, parametrized as a left perturbation of the initial guess:
`T_C_B = exp(δξ) · T_C_B⁰`. By default `δξ` is rotation only (3-DoF) and translation stays at the
prior. In the real setup that prior is the camera position measured in the machine PCD.
`--opt-trans` makes it 6-DoF.

**Features**
- **Lidar edges:** depth discontinuities along each scan ring (Levinson & Thrun). A point is kept if
  a same-ring neighbour (azimuth step < 1°) is more than 0.5 m farther away, so only the
  *foreground* side of an edge is used. Range 2–50 m, forward half only, transformed to the body
  frame with the known `T_B_L`, and cropped to points that project into the image under the initial guess.
- **Image edges:** Canny (50/150) on the Gaussian-blurred grayscale image. Precomputed per frame:
  the distance transform, the nearest edge pixel for every pixel, and that edge pixel's normal
  (from the image gradient).

**Optimization (three stages, all frames jointly)**
1. **Stage A — rotation grid search.** ±4° in 1° steps (729 candidates) on the coarsest cost level.
   The cost is non-convex, so this places the solution in the right basin.
2. **Stage B — coarse-to-fine least squares** on the truncated, blurred distance transform sampled
   bilinearly at the projected lidar edges. The levels are (truncation, blur) = (30, 6) → (15, 3) → (6, 1) px.
   Solved with `scipy.optimize.least_squares`, using numerical Jacobians.
3. **Stage C — point-to-line ICP.** Each projected lidar edge is paired with its nearest image
   edge pixel inside a gate. The residual is the signed distance to the local edge line through that
   pixel, `nᵀ(u − q)`. Huber loss; the gate shrinks 5 → 3 → 2 px with re-association each round.

**Metrics vs ground truth**
- Rotation error: angle of `R_est · R_gtᵀ`. It is also split per axis in camera coordinates:
  pitch = x (right), yaw = y (down), roll = z (optical axis).
- Translation error: distance between estimated and true camera centres.
- `px`: median distance from projected lidar edges to the nearest image edge. Ground-truth reference: **2.12 px**.

## Benchmark results

Drive `2011_09_26_drive_0005`, camera 2, 10 frames, 20 random trials per cell. Errors are injected in
a random direction at the stated magnitude. Median runtime is **0.7 s per calibration** (Apple
silicon laptop).

### 1. Rotation only (default; translation held at the prior)

Final (stage C) rotation error. Each cell is median / p90 (°) and the success rate (< 0.5°).
Translation is not optimized in this mode, so its error equals the injected value.

| Injected rotation | Translation prior error 0 | 3 cm | 10 cm |
|---|---|---|---|
| 0.5° | 0.22 / 0.23 — 100 % | 0.23 / 0.32 — 100 % | 0.41 / 0.63 — 75 % |
| 1° | 0.23 / 0.24 — 100 % | 0.23 / 0.28 — 100 % | 0.33 / 0.59 — 75 % |
| 2° | 0.23 / 0.24 — 100 % | 0.21 / 0.33 — 100 % | 0.31 / 0.46 — 100 % |
| 3° | 0.23 / 0.23 — 100 % | 0.19 / 0.26 — 100 % | 0.33 / 0.58 — 80 % |
| 4° | 0.23 / 0.24 — 100 % | 0.25 / 0.31 — 100 % | 0.31 / 0.55 — 80 % |
| 6° | 0.22 / 0.23 — 95 % | 0.28 / 0.33 — 95 % | 0.35 / 0.62 — 70 % |

**Contribution of each stage** (median rotation error, 2° injected, 3 cm prior):

| Init | Stage A | Stage B | Stage C |
|---|---|---|---|
| 2.00° | 0.74° | 0.24° | **0.21°** |

**Residual error per axis** (stage C, 2° injected, 0 cm prior; median absolute): pitch 0.11°,
yaw 0.11°, roll 0.17°.

### 2. Rotation + translation (`--opt-trans`)

Final (stage C) results. Each cell is median rotation error (°) / median translation error (m).

| Injected rotation | Injected translation 0 | 3 cm | 10 cm |
|---|---|---|---|
| 1° | 0.49 / 0.098 | 0.49 / 0.098 | 0.49 / 0.098 |
| 2° | 0.48 / 0.096 | 0.48 / 0.097 | 0.48 / 0.097 |
| 4° | 0.44 / 0.089 | 0.43 / 0.088 | 0.44 / 0.090 |

Error per axis: pitch about 0.45°, yaw 0.01°, roll 0.07°. Stage B alone ends at about 0.52° / 0.10 m.

The success rate (< 0.5°) here is 95–100 %, but only because 0.48° falls just under the threshold.
The translation error is about 10 cm in every case.

### Findings

1. **Rotation converges reliably.** For injected errors up to 4°, every trial ends at about 0.23°
   from KITTI ground truth, independent of the starting error. At 6°, 1 in 20 trials falls outside
   the ±4° grid search and fails (about 7.6°). A wider grid or a better initial guess fixes that.
2. **Translation is not observable with this cost on this data.** With 6-DoF optimization, every
   trial converges to the same pose: about 0.45° pitch and about 10 cm off, even when started from
   zero translation error. Pitch and camera height trade off along a nearly flat valley, and the
   edge-distance score at that pose (2.02 px) is *lower* than at ground truth (2.12 px).
   **Translation therefore has to come from a prior**: for the general vehicle, the camera position measured in the machine PCD.
3. **Translation prior accuracy drives rotation accuracy.** A 3 cm prior error is as good as a perfect
   prior. A 10 cm prior error raises the median to about 0.33° and the p90 to about 0.6°, and 20–30 % of
   trials miss 0.5°. **Target: camera position known to ≤ 3 cm.**
4. **Stage B does most of the work; stage C adds about 0.03°.**
5. **The remaining ~0.2° is a repeatable bias, not noise.** Likely causes, not yet separated:
   - imperfect KITTI ground truth (about 1 px in places);
   - Velodyne motion distortion (the car moves at 3–6 m/s in this drive);
   - edge-feature bias: foreground points sit just inside the true edge, and texture edges from trees and facades pull the alignment.

## Known limitations / next steps

- No lidar–camera occlusion handling. The baseline is small on KITTI but not on a general vehicle.
- No filtering of vegetation or clutter among the lidar edge points, and Canny responds to texture.
  A line-consistency filter on lidar edges and/or a learned edge detector should tighten the result.
- Stage C has no edge-direction consistency check when pairing points with image edges.
- The coarse stages depend on viewpoint diversity across frames. On the vehicle, capture 10–20
  distinct poses rather than many frames from one spot.
