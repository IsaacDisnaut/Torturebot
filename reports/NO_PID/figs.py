import pickle, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from metrics import enrich
from yawstats import add
OUT = "/home/isaac/turtlebot3/NO_PID/report/"
PREV = "/tmp/claude-1000/-home-isaac-turtlebot3-NO-PID/9e8e8a72-be95-4227-a099-d1e3459066e7/scratchpad/"
plt.rcParams.update({"font.family": "Noto Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8a8a86", "axes.labelcolor": "#3a3a38", "xtick.color": "#5a5a57", "ytick.color": "#5a5a57",
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "axes.axisbelow": True, "svg.fonttype": "path", "axes.unicode_minus": False})
# fixed categorical order (slots 1-5)
G = {"nopid": ("NO PID", "#2a78d6"), "pid": ("PID (tuned)", "#eb6834"), "pid_before": ("PID (before tune)", "#1baf7a"),
     "nopid_unb": ("NO PID unbalance", "#eda100"), "pid_unb": ("PID unbalance", "#e87ba4")}
INK = "#3a3a38"; CMD = "#5f5e5a"
runs = add(enrich(pickle.load(open("runs.pkl", "rb"))))
SP = [0.0335, 0.067, 0.134]
OFF = dict(zip(G, [-0.32, -0.16, 0.0, 0.16, 0.32]))
def sel(g, v): return [r for r in runs if r["group"] == g and abs(r["v_set"] - v) < 1e-6]
def save(fig, n):
    fig.savefig(OUT + n, bbox_inches="tight"); fig.savefig(PREV + "prev_" + n.replace(".svg", ".png"), dpi=90, bbox_inches="tight"); plt.close(fig)
def rel(t, y, ts): return y - y[max(np.searchsorted(t, ts) - 1, 0)]

def dots(ax, val):
    for g in G:
        for i, v in enumerate(SP):
            rs = sel(g, v)
            if not rs: continue
            y = [val(r) for r in rs]; x = i + OFF[g] + np.linspace(-0.045, 0.045, len(y))
            ax.scatter(x, y, s=18, color=G[g][1], edgecolor="white", linewidth=0.8, zorder=3, label=G[g][0] if i == 0 else None)
            ax.hlines(np.mean(y), i + OFF[g] - 0.07, i + OFF[g] + 0.07, color=INK, lw=2, zorder=4)
    ax.set_xticks(range(3), [f"{v} m/s" for v in SP]); ax.set_xlim(-0.5, 2.5)

# F1 steady-state speed
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
dots(axs[0], lambda r: (r["cmd_mean"] / r["v_set"] - 1) * 100)
dots(axs[1], lambda r: (r["odom_v_pose"] / r["v_set"] - 1) * 100)
for ax, ttl in zip(axs, ("cmd_vel sent (steady) vs setpoint", "Actual speed (odom pose) vs setpoint")):
    ax.axhline(0, color=INK, lw=1); ax.set_ylabel("% difference from setpoint"); ax.set_title(ttl, loc="left", fontsize=10, color=INK)
axs[0].legend(frameon=False, fontsize=7.5, loc="upper right")
fig.tight_layout(); save(fig, "f1_speed.svg")

# F2 speed time series, one representative run per cell
def pick(g, v):
    rs = [r for r in sel(g, v) if not r["late_start"] and np.nanmax(r["pose_v"][1]) < 3 * v] or sel(g, v)
    med = np.median([r["odom_v_pose"] for r in rs]); return min(rs, key=lambda r: abs(r["odom_v_pose"] - med))
def timeseries(groups, name):
    fig, axs = plt.subplots(3, 2, figsize=(9, 7.2), sharey="row")
    for i, v in enumerate(SP):
        for j, g in enumerate(groups):
            ax = axs[i, j]; r = pick(g, v); ts = r["ts"]; d = r["data"]
            c = d["/cmd_vel"]; ax.step(c[:, 0] - ts, c[:, 1], where="post", color=CMD, lw=1.5, label="cmd_vel")
            tg, vv = r["pose_v"]; ax.plot(tg - ts, vv, color=G[g][1], lw=2, label="actual (odom pose, 0.5 s window)")
            if g.startswith("pid") and "/odom_ekf" in d:
                e = d["/odom_ekf"]; ax.plot(e[:, 0] - ts, e[:, 1], color=G[g][1], lw=1, ls=(0, (2, 1.5)), label="/odom_ekf")
            ax.axhline(v, color=INK, lw=0.8, ls=":")
            ax.set_xlim(-1, r["te"] - ts + 2); ax.set_ylim(-0.005, v * 2.4)
            ax.set_title(f"{G[g][0]}  |  setpoint {v} m/s  |  {r['name']}", loc="left", fontsize=8, color=INK)
            if j == 0: ax.set_ylabel("linear.x [m/s]")
            if i == 2: ax.set_xlabel("time since motion start [s]")
            if i == 0: ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(); save(fig, name)
timeseries(["nopid", "pid"], "f2_timeseries.svg")
timeseries(["nopid_unb", "pid_unb"], "f2c_timeseries_unb.svg")

# F2b start transient of cmd_vel
fig, axs = plt.subplots(1, 3, figsize=(9, 2.9))
for i, v in enumerate(SP):
    ax = axs[i]
    for g in ["nopid", "pid_before", "pid", "pid_unb"]:
        for k, r in enumerate([r for r in sel(g, v) if not r["late_start"]]):
            c = r["data"]["/cmd_vel"]; ax.step(c[:, 0] - r["ts"], c[:, 1] / v, where="post", color=G[g][1], lw=1.1, alpha=0.85,
                                                label=G[g][0] if k == 0 else None)
    ax.set_xlim(-0.3, 3); ax.axhline(1, color=INK, lw=0.8, ls=":")
    ax.set_title(f"setpoint {v} m/s", loc="left", fontsize=9, color=INK); ax.set_xlabel("time since motion start [s]")
axs[0].set_ylabel("cmd_vel / setpoint"); axs[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); save(fig, "f2b_kick.svg")

# F3 yaw vs distance
def yaw_vs_dist(groups, name, ylim):
    fig, axs = plt.subplots(3, len(groups), figsize=(9, 7), sharex=True, sharey=True, squeeze=False)
    for i, v in enumerate(SP):
        for j, g in enumerate(groups):
            ax = axs[i, j]; rs = sel(g, v)
            if not rs:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="#8a8a86"); ax.grid(False)
            for r in rs:
                im = r["data"]["/imu"]; o = r["data"]["/odom"]
                s = rel(o[:, 0], np.concatenate([[0], np.cumsum(np.hypot(np.diff(o[:, 3]), np.diff(o[:, 4])))]), r["ts"])
                y = rel(im[:, 0], np.degrees(np.unwrap(im[:, 1])), r["ts"])
                w = (im[:, 0] >= r["ts"] - 0.5) & (im[:, 0] <= r["te"] + 1)
                ax.plot(np.interp(im[w, 0], o[:, 0], s), y[w], color=G[g][1], lw=1.3, alpha=0.9)
            ax.axhline(0, color=INK, lw=0.8); ax.set_ylim(*ylim)
            if i == 0: ax.set_title(G[g][0], loc="left", fontsize=10, color=INK)
            if j == 0: ax.set_ylabel(f"{v} m/s\nyaw (IMU) [deg]")
            if i == 2: ax.set_xlabel("distance travelled [m]")
    fig.tight_layout(); save(fig, name)
yaw_vs_dist(["nopid", "pid", "pid_before"], "f3_yaw.svg", (-1.5, 3))
yaw_vs_dist(["nopid_unb", "pid_unb"], "f3b_yaw_unb.svg", (-12.5, 4))

# F4 yaw steadiness dots
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
dots(axs[0], lambda r: r["yaw_absmax"]); dots(axs[1], lambda r: r["yaw_pp"])
for ax, ttl in zip(axs, ("Max |yaw| during run", "Yaw range during run (max - min)")):
    ax.set_title(ttl, loc="left", fontsize=10, color=INK); ax.set_ylim(0, None)
axs[0].set_ylabel("yaw [deg]"); axs[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); save(fig, "f4_yaw_end.svg")

# F5 heading sources at 0.134 m/s unbalanced: IMU vs wheel encoders vs EKF
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, g in zip(axs, ["nopid_unb", "pid_unb"]):
    for r in sel(g, 0.134):
        ts = r["ts"]; im = r["data"]["/imu"]; w = r["data"]["/wheel"]
        ax.plot(im[:, 0] - ts, rel(im[:, 0], np.degrees(np.unwrap(im[:, 1])), ts), color=G[g][1], lw=1.5)
        ax.plot(w[:, 0] - ts, rel(w[:, 0], w[:, 1], ts), color=CMD, lw=1, ls=(0, (3, 2)))
        if g.startswith("pid"):
            e = r["data"]["/odom_ekf"]; ax.plot(e[:, 0] - ts, rel(e[:, 0], np.degrees(np.unwrap(e[:, 5])), ts), color=INK, lw=1, ls=":")
    ax.axhline(0, color=INK, lw=0.8); ax.set_xlim(-0.5, 11); ax.set_xlabel("time since motion start [s]")
    ax.set_title(f"{G[g][0]}  |  0.134 m/s  |  5 runs", loc="left", fontsize=9.5, color=INK)
axs[0].set_ylabel("yaw [deg]  (+ = left / CCW)")
axs[1].legend(handles=[Line2D([], [], color=G["pid_unb"][1], lw=1.5, label="IMU (actual body rotation)"),
                       Line2D([], [], color=CMD, lw=1, ls=(0, (3, 2)), label="wheel encoders (r·(θR−θL)/L)"),
                       Line2D([], [], color=INK, lw=1, ls=":", label="/odom_ekf")], frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout(); save(fig, "f5_heading_sources.svg")
print("ok")

# F6 yaw steadiness split: drift vs fluctuation
from yawstats import steadiness
for r in runs: r.update({"st_" + k: v for k, v in steadiness(r).items()})
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
dots(axs[0], lambda r: r["st_drift"]); dots(axs[1], lambda r: r["st_res_sd"])
axs[0].axhline(0, color=INK, lw=1)
axs[0].set_title("Yaw drift (slope of yaw vs distance)", loc="left", fontsize=10, color=INK); axs[0].set_ylabel("deg per metre  (+ = left)")
axs[1].set_title("Yaw fluctuation (SD after removing drift)", loc="left", fontsize=10, color=INK); axs[1].set_ylabel("deg"); axs[1].set_ylim(0, None)
axs[0].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.tight_layout(); save(fig, "f6_yaw_steadiness.svg")

# F7 detrended yaw at 0.134 m/s: what the fluctuation looks like
fig, axs = plt.subplots(1, 4, figsize=(9, 2.8), sharey=True)
for ax, g in zip(axs, ["nopid", "nopid_unb", "pid", "pid_unb"]):
    for r in sel(g, 0.134):
        im = r["data"]["/imu"]; o = r["data"]["/odom"]
        y = rel(im[:, 0], np.degrees(np.unwrap(im[:, 1])), r["ts"])
        s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(o[:, 3]), np.diff(o[:, 4])))])
        w = (im[:, 0] >= r["ts"] + 1.0) & (im[:, 0] <= r["te"] - 0.5); d = np.interp(im[w, 0], o[:, 0], s); d = d - d[0]
        k, b = np.polyfit(d, y[w], 1); ax.plot(d, y[w] - (k * d + b), color=G[g][1], lw=1.1, alpha=0.9)
    ax.axhline(0, color=INK, lw=0.8); ax.set_title(G[g][0], loc="left", fontsize=9, color=INK); ax.set_xlabel("distance [m]")
axs[0].set_ylabel("yaw minus trend [deg]")
fig.tight_layout(); save(fig, "f7_yaw_detrended.svg")
print("ok2")
