import rosbag2_py, glob, os, numpy as np, pickle, sys
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
R_W, SEP = 0.033, 0.160
def wheel_yaw(bagdir):
    mc = glob.glob(bagdir + "/**/*.mcap", recursive=True)[0]
    r = rosbag2_py.SequentialReader(); r.open(rosbag2_py.StorageOptions(uri=os.path.dirname(mc), storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    ty = {t.name: t.type for t in r.get_all_topics_and_types()}; r.set_filter(rosbag2_py.StorageFilter(topics=["/joint_states"]))
    out = []
    while r.has_next():
        tp, raw, t = r.read_next(); m = deserialize_message(raw, get_message(ty[tp]))
        p = dict(zip(m.name, m.position)); out.append((t * 1e-9, p["wheel_left_joint"], p["wheel_right_joint"]))
    a = np.array(out); return np.column_stack([a[:, 0], np.degrees(R_W * (a[:, 2] - a[:, 1]) / SEP), R_W * (a[:, 1] + a[:, 2]) / 2])
if __name__ == "__main__":
    runs = pickle.load(open("runs.pkl", "rb")); ROOT = "/home/isaac/turtlebot3/NO_PID/"
    for r in runs:
        d = ROOT + ("ZBefore_Tune/" if r["group"] == "pid_before" else "") + r["name"]
        w = wheel_yaw(d); r["data"]["/wheel"] = w
        i0 = max(np.searchsorted(w[:, 0], r["ts"]) - 1, 0); after = np.searchsorted(w[:, 0], r["te"] + 0.5)
        after = min(after, len(w) - 1)
        r["yaw_wheel_end"] = w[after, 1] - w[i0, 1]; r["dist_wheel"] = w[after, 2] - w[i0, 2]
        if r["v_set"] == 0.134 or "unb" in r["group"]:
            print(f'{r["group"]:10s} {r["name"]:34s} IMU={r["yaw_imu_end"]:+6.2f} wheel={r["yaw_wheel_end"]:+6.2f} EKF={r.get("yaw_ekf_end", np.nan):+6.2f} dist_w={r["dist_wheel"]:.3f}')
    pickle.dump(runs, open("runs.pkl", "wb"))
