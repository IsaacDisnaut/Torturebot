import sys, os, re, glob, pickle
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

ROOT = sys.argv[1]
TOP = ["/cmd_vel", "/odom", "/odom_ekf", "/imu"]

def yaw(q): return np.arctan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

def read(bagdir):
    mcap = glob.glob(os.path.join(bagdir, "**", "*.mcap"), recursive=True)[0]
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=os.path.dirname(mcap), storage_id="mcap"),
           rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in TOP if t in types]))
    d = {k: [] for k in TOP}
    while r.has_next():
        tp, raw, t = r.read_next()
        m = deserialize_message(raw, get_message(types[tp]))
        t *= 1e-9
        if tp == "/cmd_vel":
            tw = m.twist if hasattr(m, "header") else m
            d[tp].append((t, tw.linear.x, tw.angular.z))
        elif tp == "/imu":
            d[tp].append((t, yaw(m.orientation), m.angular_velocity.z))
        else:
            p = m.pose.pose
            d[tp].append((t, m.twist.twist.linear.x, m.twist.twist.angular.z, p.position.x, p.position.y, yaw(p.orientation)))
    return {k: np.array(v) for k, v in d.items() if v}

def parse(name, before):
    m = re.match(r"(nopid|pid)_([\d.]+)m_([\d.]+)v_?(.*)$", name)
    ctrl, dist, v, rest = m.groups()
    group = "pid_before" if before else ctrl + ("_unb" if "unbalance" in rest else "")
    return dict(group=group, dist=float(dist), v_set=float(v), label=rest)

runs = []
dirs = [(d, False) for d in sorted(glob.glob(os.path.join(ROOT, "*pid_*")))] + \
       [(d, True) for d in sorted(glob.glob(os.path.join(ROOT, "ZBefore_Tune", "*pid_*")))]
for d, before in dirs:
    if not os.path.isdir(d): continue
    name = os.path.basename(d)
    info = parse(name, before); info["name"] = name
    data = read(d)
    cmd = data["/cmd_vel"]
    mv = cmd[np.abs(cmd[:, 1]) > 1e-4]
    ts, te = mv[0, 0], mv[-1, 0]
    info["t_move"] = te - ts
    st = lambda a: (a[:, 0] >= ts + 1.5) & (a[:, 0] <= te - 0.5)   # steady window
    c = cmd[st(cmd)]
    info["cmd_mean"] = c[:, 1].mean(); info["cmd_std"] = c[:, 1].std()
    info["cmd_max"] = cmd[:, 1].max(); info["cmd_wz_absmax"] = np.abs(cmd[:, 2]).max()
    info["cmd_wz_mean"] = c[:, 2].mean()
    o = data["/odom"]; os_ = o[st(o)]
    info["odom_v_mean"] = os_[:, 1].mean(); info["odom_v_std"] = os_[:, 1].std()
    # avg speed from pose displacement over steady window
    dx = np.hypot(np.diff(os_[:, 3]), np.diff(os_[:, 4])).sum()
    info["odom_v_pose"] = dx / (os_[-1, 0] - os_[0, 0])
    # yaw: relative to yaw just before motion
    def yaw_stats(arr, col):
        y = np.unwrap(arr[:, col])
        i0 = np.searchsorted(arr[:, 0], ts) - 1; i0 = max(i0, 0)
        y = np.degrees(y - y[i0])
        w = (arr[:, 0] >= ts) & (arr[:, 0] <= te + 1.0)
        after = arr[:, 0] >= te + 0.5
        return y[after][0] if after.any() else y[w][-1], y[w][np.argmax(np.abs(y[w]))]
    info["yaw_imu_end"], info["yaw_imu_maxabs"] = yaw_stats(data["/imu"], 1)
    info["yaw_odom_end"], info["yaw_odom_maxabs"] = yaw_stats(o, 5)
    if "/odom_ekf" in data:
        e = data["/odom_ekf"]; es = e[st(e)]
        info["ekf_v_mean"] = es[:, 1].mean(); info["ekf_v_std"] = es[:, 1].std()
        info["yaw_ekf_end"], info["yaw_ekf_maxabs"] = yaw_stats(e, 5)
    # lateral drift in odom frame relative to start heading
    i0 = max(np.searchsorted(o[:, 0], ts) - 1, 0)
    info["ts"] = ts; info["te"] = te
    info["data"] = data
    runs.append(info)
    print(f'{info["group"]:10s} {info["v_set"]:.4f} {name:32s} cmd={info["cmd_mean"]:.4f} odom={info["odom_v_mean"]:.4f} pose={info["odom_v_pose"]:.4f} '
          f'ekf={info.get("ekf_v_mean",np.nan):.4f} yawIMU={info["yaw_imu_end"]:+.2f} yawOdom={info["yaw_odom_end"]:+.2f} yawEKF={info.get("yaw_ekf_end",np.nan):+.2f} wz|max|={info["cmd_wz_absmax"]:.3f} T={info["t_move"]:.1f}')
pickle.dump(runs, open(sys.argv[2], "wb"))
