import numpy as np
def yaw_run(r):
    im = r["data"]["/imu"]; y = np.degrees(np.unwrap(im[:, 1]))
    j0 = max(np.searchsorted(im[:, 0], r["ts"]) - 1, 0); y = y - y[j0]
    w = (im[:, 0] >= r["ts"]) & (im[:, 0] <= r["te"]); return y[w]
def add(runs):
    for r in runs:
        y = yaw_run(r); r["yaw_sd"] = y.std(); r["yaw_pp"] = y.max() - y.min(); r["yaw_absmax"] = np.abs(y).max()
    return runs

def steadiness(r):
    """drift (deg/m, linear fit of yaw vs distance), residual SD after removing the trend, SD of gyro z (deg/s), sign changes of yaw rate"""
    im = r["data"]["/imu"]; o = r["data"]["/odom"]
    y = np.degrees(np.unwrap(im[:, 1])); y = y - y[max(np.searchsorted(im[:, 0], r["ts"]) - 1, 0)]
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(o[:, 3]), np.diff(o[:, 4])))])
    w = (im[:, 0] >= r["ts"] + 1.0) & (im[:, 0] <= r["te"] - 0.5)          # skip start transient
    d = np.interp(im[w, 0], o[:, 0], s); yy = y[w]
    k, b = np.polyfit(d, yy, 1); res = yy - (k * d + b)
    gz = np.degrees(im[w, 2])
    return dict(drift=k, res_sd=res.std(), gz_sd=gz.std())
