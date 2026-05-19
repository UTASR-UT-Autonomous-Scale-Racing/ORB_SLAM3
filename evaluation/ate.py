#!/usr/bin/env python3
"""ATE RMSE of an EuRoC trajectory (SaveTrajectoryEuRoC output) against
evaluation/Ground_truth/EuRoC_left_cam, after SE(3) (stereo) or Sim(3)
(monocular, --scale) Umeyama alignment. Python 3 port of evaluate_ate_scale.py.

    python3 evaluation/ate.py GT.txt estimate.txt [--scale]
"""
import argparse
import numpy as np


def load(path):
    rows = [l.replace(",", " ").split() for l in open(path) if l.strip() and not l.startswith("#")]
    data = np.array([[float(v) for v in r[:4]] for r in rows])
    t = data[:, 0]
    if t[0] > 1e14:          # nanoseconds
        t = t * 1e-9
    return t, data[:, 1:4]


def associate(t_gt, t_est, max_dt=0.02):
    idx = np.searchsorted(t_gt, t_est)
    idx = np.clip(idx, 1, len(t_gt) - 1)
    prev = idx - 1
    nearest = np.where(np.abs(t_gt[prev] - t_est) < np.abs(t_gt[idx] - t_est), prev, idx)
    keep = np.abs(t_gt[nearest] - t_est) < max_dt
    return nearest[keep], np.flatnonzero(keep)


def umeyama(model, data, scale):
    """R, t, s minimising |s R data + t - model|."""
    mu_m, mu_d = model.mean(1, keepdims=True), data.mean(1, keepdims=True)
    m, d = model - mu_m, data - mu_d
    U, D, Vt = np.linalg.svd(m @ d.T)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (d ** 2).sum() if scale else 1.0
    t = mu_m - s * R @ mu_d
    return R, t, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gt")
    ap.add_argument("est")
    ap.add_argument("--scale", action="store_true", help="Sim(3) alignment (monocular)")
    a = ap.parse_args()
    t_gt, p_gt = load(a.gt)
    t_est, p_est = load(a.est)
    i_gt, i_est = associate(t_gt, t_est)
    model, data = p_gt[i_gt].T, p_est[i_est].T
    R, t, s = umeyama(model, data, a.scale)
    err = np.linalg.norm(s * R @ data + t - model, axis=0)
    print(f"ate_rmse {np.sqrt((err ** 2).mean()):.4f} m  pairs {len(err)}  scale {s:.4f}")


if __name__ == "__main__":
    main()
