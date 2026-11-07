import pickle, numpy as np, pandas as pd

def pose_speed(o, win=0.5):
    t = o[:, 0]; s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(o[:, 3]), np.diff(o[:, 4])))])
    tg = np.arange(t[0], t[-1], 0.02); sg = np.interp(tg, t, s)
    n = int(win / 0.02); v = np.full_like(tg, np.nan)
    v[n:-n] = (sg[2*n:] - sg[:-2*n]) / (2 * n * 0.02)
    return tg, v

def enrich(runs):
    for r in runs:
        # start transient not fully recorded: PID ramp already running at first cmd, or odom starts after motion began
        r["late_start"] = (r["group"].startswith("pid") and abs(r["data"]["/cmd_vel"][0, 1]) > 1e-4) or r["data"]["/odom"][0, 0] > r["ts"] - 0.2
        tg, v = pose_speed(r["data"]["/odom"]); r["pose_v"] = (tg, v)
        vs = r["v_set"]
        w = (tg >= r["ts"]) & (tg <= r["ts"] + 6)
        if not r["late_start"]:
            r["meas_peak_ratio"] = np.nanmax(v[w]) / vs
            # settle: first time after which v stays within ±5% of the steady mean until te-0.5
            st = r["odom_v_pose"]; band = 0.05 * vs
            ww = (tg >= r["ts"]) & (tg <= r["te"] - 0.5)
            out = np.where(np.abs(v[ww] - st) > band)[0]
            r["t_settle"] = (tg[ww][out[-1]] - r["ts"]) if len(out) else 0.0
            r["cmd_peak_ratio"] = r["cmd_max"] / vs
        else:
            r["meas_peak_ratio"] = r["t_settle"] = r["cmd_peak_ratio"] = np.nan
        r["err_pct"] = (r["odom_v_pose"] / vs - 1) * 100
    return runs

def table(runs):
    rows = []
    for r in runs:
        rows.append({k: r.get(k, np.nan) for k in ["group","v_set","name","cmd_mean","odom_v_pose","odom_v_mean","ekf_v_mean","err_pct",
            "yaw_imu_end","yaw_imu_maxabs","yaw_ekf_end","cmd_wz_absmax","cmd_peak_ratio","meas_peak_ratio","t_settle","t_move","late_start"]})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    runs = enrich(pickle.load(open("runs.pkl", "rb")))
    df = table(runs); df.to_csv("runs.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    g = df.groupby(["group", "v_set"])
    s = pd.DataFrame({
        "n": g.size(), "cmd": g.cmd_mean.mean(), "v_real": g.odom_v_pose.mean(), "v_real_sd": g.odom_v_pose.std(),
        "err%": g.err_pct.mean(), "ekf": g.ekf_v_mean.mean(), "odomTw": g.odom_v_mean.mean(),
        "yawEnd": g.yaw_imu_end.mean(), "yawEnd_sd": g.yaw_imu_end.std(), "|yawEnd|": g.yaw_imu_end.apply(lambda x: x.abs().mean()),
        "|yaw|max": g.yaw_imu_maxabs.apply(lambda x: x.abs().mean()), "wz_max": g.cmd_wz_absmax.max(),
        "cmdPk": g.cmd_peak_ratio.mean(), "measPk": g.meas_peak_ratio.mean(), "tSettle": g.t_settle.mean(), "T": g.t_move.mean()})
    print(s.round(4).to_string())
    print(df[["name","yaw_imu_end","yaw_imu_maxabs","meas_peak_ratio","t_settle"]].round(3).to_string())
