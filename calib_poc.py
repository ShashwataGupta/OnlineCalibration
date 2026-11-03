#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "opencv-python-headless"]
# ///
"""Targetless camera-to-body extrinsic refinement POC on KITTI raw.

The lidar-to-body transform is treated as known (KITTI IMU = "body"), the
camera-to-body transform is perturbed and then recovered by aligning lidar
depth-discontinuity edges with image edges (truncated distance transform),
in three stages: (A) rotation grid search, (B) coarse-to-fine least squares on the
blurred distance transform, (C) point-to-line refinement against local image edge lines.

    uv run calib_poc.py download --root ~/data/kitti      # ~650 MB
    uv run calib_poc.py run --root ~/data/kitti --trials 5 --rot-deg 2 --trans-m 0.03
    uv run calib_poc.py bench --root ~/data/kitti --trials 10   # error sweep -> bench.csv

Overlays (red = image edges, green = projected lidar edges) go to --out.
"""

import argparse
import csv
import itertools
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
LEVELS = [(30.0, 6.0), (15.0, 3.0), (6.0, 1.0)]  # (truncation px, blur sigma px)


# ---------------------------------------------------------------- data
def download(root: Path, date: str, drive: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, target in [
        (f"{date}_calib.zip", root / date / "calib_cam_to_cam.txt"),
        (
            f"{date}_drive_{drive}/{date}_drive_{drive}_sync.zip",
            root / date / f"{date}_drive_{drive}_sync",
        ),
    ]:
        if target.exists():
            continue
        zp = root / Path(name).name
        print(f"downloading {name}")
        urllib.request.urlretrieve(URL + name, zp)
        with zipfile.ZipFile(zp) as z:
            z.extractall(root)
        zp.unlink()


def read_calib(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        key, _, val = line.partition(":")
        try:
            out[key] = np.array(val.split(), float)
        except ValueError:
            pass
    return out


def rt(d: dict) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = d["R"].reshape(3, 3), d["T"]
    return T


def lidar_edges(pts: np.ndarray, min_jump: float = 0.5) -> np.ndarray:
    """Foreground points at range discontinuities along each scan ring (Levinson & Thrun)."""
    r = np.linalg.norm(pts, axis=1)
    az = np.arctan2(pts[:, 1], pts[:, 0])
    score = np.zeros_like(r)
    for s in (-1, 1):  # neighbour must be on the same ring (small azimuth step) and farther away
        same_ring = np.abs(np.angle(np.exp(1j * (np.roll(az, s) - az)))) < np.deg2rad(1.0)
        score = np.maximum(score, np.where(same_ring, np.roll(r, s) - r, 0))
    keep = (score > min_jump) & (r > 2) & (r < 50) & (pts[:, 0] > 0)
    return pts[keep]


def load(root: Path, date: str, drive: str, cam: int, n_frames: int, max_speed=np.inf):
    c = root / date
    T_L_B = rt(read_calib(c / "calib_imu_to_velo.txt"))  # p_L = T_L_B p_B
    T_C_L = rt(read_calib(c / "calib_velo_to_cam.txt"))  # cam0, unrectified
    cc = read_calib(c / "calib_cam_to_cam.txt")
    R_rect = np.eye(4)
    R_rect[:3, :3] = cc["R_rect_00"].reshape(3, 3)
    P = cc[f"P_rect_0{cam}"].reshape(3, 4) @ R_rect  # cam0 frame -> rectified pixels of `cam`
    T_B_L = np.linalg.inv(T_L_B)

    d = c / f"{date}_drive_{drive}_sync"
    bins = sorted((d / "velodyne_points/data").glob("*.bin"))
    speed = np.array(
        [np.linalg.norm(np.loadtxt(d / f"oxts/data/{b.stem}.txt")[8:11]) for b in bins]
    )
    bins = [b for b, v in zip(bins, speed) if v <= max_speed]  # e.g. static frames only
    speed = speed[speed <= max_speed]
    sel = np.linspace(0, len(bins) - 1, n_frames).astype(int)
    print(
        f"drive {drive}: {len(bins)} frames <= {max_speed} m/s, using {n_frames},"
        f" speed m/s min/median/max {speed[sel].min():.2f}/{np.median(speed[sel]):.2f}"
        f"/{speed[sel].max():.2f}"
    )
    frames = []
    for i in sel:
        pts = np.fromfile(bins[i], np.float32).reshape(-1, 4)[:, :3].astype(float)
        E = lidar_edges(pts)
        X_B = E @ T_B_L[:3, :3].T + T_B_L[:3, 3]
        img = cv2.imread(str(d / f"image_0{cam}/data/{bins[i].stem}.png"), cv2.IMREAD_GRAYSCALE)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        dt, lab = cv2.distanceTransformWithLabels(
            255 - edges, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        ey, ex = np.nonzero(edges)  # row-major, matches the 1-based pixel label order
        n = np.c_[
            cv2.Sobel(blur, cv2.CV_32F, 1, 0)[ey, ex], cv2.Sobel(blur, cv2.CV_32F, 0, 1)[ey, ex]
        ]
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9  # edge normal = gradient direction
        frames.append(
            {
                "img": img,
                "edges": edges,
                "dt": dt,
                "X": X_B,
                "near": np.c_[ex, ey][lab - 1].astype(float),  # nearest edge pixel (u, v) per pixel
                "normal": n[lab - 1],  # normal of that nearest edge pixel
            }
        )
    return frames, P, T_C_L @ T_L_B


def load_drives(a):
    """Concatenate frames from comma-separated drives of one date (they share one calibration)."""
    loaded = [load(a.root, a.date, d, a.cam, a.frames, a.max_speed) for d in a.drive.split(",")]
    return [f for fr, _, _ in loaded for f in fr], loaded[0][1], loaded[0][2]


# ---------------------------------------------------------------- model
def se3(x: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rotation.from_rotvec(x[:3]).as_matrix(), x[3:]
    return T


def project(P, T_C_B, X):
    Xc = X @ T_C_B[:3, :3].T + T_C_B[:3, 3]
    uvw = Xc @ P[:, :3].T + P[:, 3]
    z = uvw[:, 2]
    return uvw[:, :2] / np.maximum(z, 1e-6)[:, None], z


def residuals(x, frames, maps, P, T0, tau):
    T = se3(x) @ T0
    out = []
    for f, m in zip(frames, maps):
        uv, z = project(P, T, f["X"])
        r = map_coordinates(m, [uv[:, 1], uv[:, 0]], order=1, mode="constant", cval=tau)
        out.append(np.where(z > 0.5, r, tau))
    return np.concatenate(out)


def point_to_line(frames, P, T0, x, full, gates=(5.0, 3.0, 2.0)):
    """Stage C: associate each projected lidar edge with its nearest image edge pixel within a
    shrinking gate and minimise the signed distance to that pixel's local edge line."""
    for gate in gates:
        T = se3(full(x)) @ T0
        sets = []
        for f in frames:
            uv, z = project(P, T, f["X"])
            u, v = np.round(uv).astype(int).T
            h, w = f["dt"].shape
            ok = (z > 0.5) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            ok[ok] = f["dt"][v[ok], u[ok]] < gate
            sets.append((f["X"][ok], f["near"][v[ok], u[ok]], f["normal"][v[ok], u[ok]]))

        def fun(x):
            T = se3(full(x)) @ T0
            return np.concatenate([((project(P, T, X)[0] - q) * n).sum(1) for X, q, n in sets])

        x = least_squares(fun, x, loss="huber", f_scale=1.0).x
    return x


def calibrate(frames, P, T0, opt_trans=False, grid_deg=4.0, grid_step=1.0) -> dict:
    """Translation is weakly observable (pitch <-> height valley), so by default it stays at
    the prior (e.g. measured from the machine PCD) and only rotation is refined.
    Returns the estimate after each stage: {"A": T, "B": T, "C": T}."""
    full = (lambda x: x) if opt_trans else (lambda x: np.r_[x, 0, 0, 0])

    def cost_maps(tau, sigma):
        return [cv2.GaussianBlur(np.minimum(f["dt"], tau), (0, 0), sigma) for f in frames]

    # Stage A: rotation-only grid search on the coarsest level (cost is non-convex)
    tau, sigma = LEVELS[0]
    maps = cost_maps(tau, sigma)
    g = np.deg2rad(np.arange(-grid_deg, grid_deg + 1e-9, grid_step))
    cands = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
    scores = [residuals(np.r_[w, 0, 0, 0], frames, maps, P, T0, tau).sum() for w in cands]
    x = np.r_[cands[int(np.argmin(scores))], 0, 0, 0][: 6 if opt_trans else 3]
    out = {"A": se3(full(x)) @ T0}

    # Stage B: coarse-to-fine least squares
    for tau, sigma in LEVELS:
        maps = cost_maps(tau, sigma)
        fun = lambda x: residuals(full(x), frames, maps, P, T0, tau)  # noqa: E731
        x = least_squares(fun, x, diff_step=1e-3).x
    out["B"] = se3(full(x)) @ T0

    # Stage C: point-to-line ICP
    out["C"] = se3(full(point_to_line(frames, P, T0, x, full))) @ T0
    return out


# ---------------------------------------------------------------- eval
def pose_error(T, T_gt):
    """(rotation error deg, camera-centre error m, rotation error rotvec deg in camera axes)."""
    w = np.rad2deg(Rotation.from_matrix(T[:3, :3] @ T_gt[:3, :3].T).as_rotvec())
    centre = lambda M: -M[:3, :3].T @ M[:3, 3]  # noqa: E731
    return np.linalg.norm(w), np.linalg.norm(centre(T) - centre(T_gt)), w


def perturb(rng, T_gt, rot_deg, trans_m):
    unit = lambda: (v := rng.normal(size=3)) / np.linalg.norm(v)  # noqa: E731
    return se3(np.r_[unit() * np.deg2rad(rot_deg), unit() * trans_m]) @ T_gt


def median_px(frames, P, T):
    return np.median(residuals(np.zeros(6), frames, [f["dt"] for f in frames], P, T, 1e3))


def overlay(f, P, T):
    img = cv2.cvtColor(f["img"], cv2.COLOR_GRAY2BGR) // 2
    img[f["edges"] > 0] = (0, 0, 255)
    uv, z = project(P, T, f["X"])
    for u, v in uv[z > 0.5].astype(int):
        cv2.circle(img, (u, v), 1, (0, 255, 0), -1)
    return img


def crop_to_view(frames, P, T, margin=40):
    """Keep only lidar edges that project into the image under the initial guess T."""
    out = []
    for f in frames:
        uv, z = project(P, T, f["X"])
        h, w = f["img"].shape
        ok = (
            (z > 0.5)
            & (uv[:, 0] > -margin)
            & (uv[:, 0] < w + margin)
            & (uv[:, 1] > -margin)
            & (uv[:, 1] < h + margin)
        )
        out.append({**f, "X": f["X"][ok]})
    return out


def run(a) -> None:
    all_frames, P, T_gt = load_drives(a)
    frames = crop_to_view(all_frames, P, T_gt, margin=0)
    print(f"{len(frames)} frames, {sum(len(f['X']) for f in frames)} lidar edge points in view")
    print(f"median edge distance @ GT: {median_px(frames, P, T_gt):.2f}px\n")
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    print("trial | init rot° trans m  px | est rot° trans m  px")
    for t in range(a.trials):
        T0 = perturb(rng, T_gt, a.rot_deg, a.trans_m)
        T = calibrate(crop_to_view(all_frames, P, T0), P, T0, a.opt_trans)["B" if a.no_p2l else "C"]
        e0, e1 = pose_error(T0, T_gt), pose_error(T, T_gt)
        print(
            f"{t:5d} | {e0[0]:8.2f} {e0[1]:7.3f} {median_px(frames, P, T0):4.1f}"
            f" | {e1[0]:7.2f} {e1[1]:7.3f} {median_px(frames, P, T):4.1f}"
        )
        cv2.imwrite(
            str(a.out / f"trial{t}.png"),
            np.vstack([overlay(frames[0], P, M) for M in (T0, T, T_gt)]),
        )
    print(f"\noverlays (init / estimate / GT) in {a.out}")


def bench(a) -> None:
    """Sweep injected camera errors; compare every stage's estimate against KITTI GT."""
    all_frames, P, T_gt = load_drives(a)
    gt_frames = crop_to_view(all_frames, P, T_gt, margin=0)
    rng, rows = np.random.default_rng(a.seed), []
    for rot, tr, t in itertools.product(a.rot_list, a.trans_list, range(a.trials)):
        T0 = perturb(rng, T_gt, rot, tr)
        t0 = time.perf_counter()
        est = calibrate(crop_to_view(all_frames, P, T0), P, T0, a.opt_trans)
        secs = time.perf_counter() - t0
        for stage, T in {"init": T0, **est}.items():
            r, c, w = pose_error(T, T_gt)
            rows.append({
                "rot_in": rot, "trans_in": tr, "trial": t, "stage": stage, "rot_err": r,
                "pitch": w[0], "yaw": w[1], "roll": w[2], "trans_err": c,
                "px": median_px(gt_frames, P, T), "secs": secs,
            })  # fmt: skip
        print(f"rot {rot}° trans {tr}m trial {t}: {rows[-1]['rot_err']:.2f}° ({secs:.1f}s)")

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "bench.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nGT reference: median edge distance {median_px(gt_frames, P, T_gt):.2f}px")
    print(f"success = rotation error < {a.success_deg}°\n")
    print(
        " rot_in trans_in stage |  med°  p90°  max° | |pitch| |yaw| |roll| | trans m |  px | ok %"
    )
    for rot, tr in itertools.product(a.rot_list, a.trans_list):
        for stage in ("init", "A", "B", "C"):
            g = [x for x in rows if (x["rot_in"], x["trans_in"], x["stage"]) == (rot, tr, stage)]
            e = np.array([x["rot_err"] for x in g])
            ax = [np.median(np.abs([x[k] for x in g])) for k in ("pitch", "yaw", "roll")]
            print(
                f"{rot:7.1f} {tr:8.2f} {stage:>5} | {np.median(e):5.2f} {np.percentile(e, 90):5.2f}"
                f" {e.max():5.2f} | {ax[0]:7.2f} {ax[1]:5.2f} {ax[2]:6.2f} |"
                f" {np.median([x['trans_err'] for x in g]):7.3f} |"
                f" {np.median([x['px'] for x in g]):4.1f} | {100 * np.mean(e < a.success_deg):4.0f}"
            )
    print(
        f"\nmedian runtime {np.median([x['secs'] for x in rows]):.1f}s; rows in {a.out}/bench.csv"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("cmd", choices=["download", "run", "bench"])
    p.add_argument("--root", type=Path, default=Path.home() / "data/kitti")
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0005", help="drive id(s), comma-separated")
    p.add_argument("--cam", type=int, default=2, help="KITTI camera index (2 = left colour)")
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--max-speed", type=float, default=np.inf, help="only frames <= this m/s")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--rot-deg", type=float, default=2.0)
    p.add_argument(
        "--trans-m",
        type=float,
        default=0.03,
        help="translation prior error (e.g. from machine PCD)",
    )
    p.add_argument(
        "--opt-trans", action="store_true", help="also refine translation (weakly observable)"
    )
    p.add_argument("--no-p2l", action="store_true", help="skip the point-to-line stage C")
    floats = lambda s: [float(v) for v in s.split(",")]  # noqa: E731
    p.add_argument("--rot-list", type=floats, default="0.5,1,2,3,4,6", help="bench: rot errors °")
    p.add_argument("--trans-list", type=floats, default="0,0.03,0.1", help="bench: trans errors m")
    p.add_argument("--success-deg", type=float, default=0.5, help="bench: success threshold")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("calib_poc_out"))
    a = p.parse_args()
    {
        "download": lambda a: [download(a.root, a.date, d) for d in a.drive.split(",")],
        "run": run,
        "bench": bench,
    }[a.cmd](a)
