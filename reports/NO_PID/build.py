import pickle, numpy as np, html
from metrics import enrich
from yawstats import add, steadiness
R = "/home/isaac/turtlebot3/NO_PID/report/"
runs = add(enrich(pickle.load(open("runs.pkl", "rb"))))
GN = {"nopid": "NO PID", "pid": "PID (หลังจูน)", "pid_before": "PID (ก่อนจูน)",
      "nopid_unb": "NO PID unbalance", "pid_unb": "PID unbalance"}
ORDER = ["nopid", "nopid_unb", "pid", "pid_unb", "pid_before"]
SP = [0.0335, 0.067, 0.134]
def sel(g, v=None): return [r for r in runs if r["group"] == g and (v is None or abs(r["v_set"] - v) < 1e-6)]
def svg(n): return open(R + n).read().split("?>", 1)[-1]
f = lambda x, d=4: f"{x:.{d}f}"
sg = lambda x, d=2: f"{x:+.{d}f}"
M = lambda rs, k: np.mean([r[k] for r in rs])

def cmd_steady(r):
    c = r["data"]["/cmd_vel"]; m = (c[:, 0] >= r["ts"] + 1.5) & (c[:, 0] <= r["te"] - 0.5); return c[m, 1]

for r in runs:
    o = r["data"]["/odom"]; i0 = max(np.searchsorted(o[:, 0], r["ts"]) - 1, 0); h = o[i0, 5]
    d = o[-1, 3:5] - o[i0, 3:5]; r["lateral_cm"] = (-np.sin(h) * d[0] + np.cos(h) * d[1]) * 100
    m = (o[:, 0] >= r["ts"]) & (o[:, 0] <= r["te"]); r["odom_gap"] = np.diff(o[m, 0]).max() if m.sum() > 1 else 0

rows_speed = ""; rows_yaw = ""
for g in ORDER:
    for v in SP:
        rs = sel(g, v)
        if not rs: continue
        cm = M(rs, "cmd_mean"); va = [r["odom_v_pose"] for r in rs]
        nl = [r for r in rs if not r["late_start"]]
        pk = np.mean([r["cmd_peak_ratio"] for r in nl]); T = np.mean([r["t_move"] for r in nl])
        cs = np.mean([cmd_steady(r).std() / v * 100 for r in rs])
        ek = [r["ekf_v_mean"] for r in rs if "ekf_v_mean" in r and g.startswith("pid")]
        rows_speed += (f"<tr><td>{GN[g]}</td><td>{v}</td><td>{len(rs)}</td><td>{f(cm)} <span class=m>({sg((cm/v-1)*100,1)}%)</span></td>"
                       f"<td><b>{f(np.mean(va))}</b> ± {f(np.std(va, ddof=1))}</td><td><b>{sg((np.mean(va)/v-1)*100,1)}%</b></td>"
                       f"<td>{f(np.mean(ek)) if ek else '–'}</td><td>{pk:.2f}×</td><td>{cs:.1f}%</td><td>{T:.1f}</td></tr>")
        yam = [r["yaw_absmax"] for r in rs]; ye = [r["yaw_imu_end"] for r in rs]
        hl = " class=hl" if max(yam) > 3 else ""
        rows_yaw += (f"<tr{hl}><td>{GN[g]}</td><td>{v}</td><td>{len(rs)}</td><td><b>{M(rs, 'yaw_sd'):.2f}</b></td><td>{M(rs, 'yaw_pp'):.2f}</td>"
                     f"<td>{np.mean(yam):.2f}</td><td><b>{max(yam):.2f}</b></td><td>{sg(np.mean(ye))} ± {np.std(ye, ddof=1):.2f}</td></tr>")

bal = [r for r in runs if not r["group"].endswith("_unb")]
yam_bal = [r["yaw_absmax"] for r in bal]; ymax_bal = max(yam_bal); n15_bal = sum(x <= 1.5 for x in yam_bal)
left_np = sum(r["yaw_imu_end"] > 0 for r in sel("nopid"))
nu = sel("nopid_unb", 0.134); pu = sel("pid_unb", 0.134)
nu_end = M(nu, "yaw_imu_end"); nu_lat = M(nu, "lateral_cm"); pu_lat = [abs(r["lateral_cm"]) for r in pu]
nu_rate = np.mean([r["yaw_imu_end"] / r["t_move"] for r in nu])
lowspd_unb = [r["yaw_absmax"] for g in ("nopid_unb", "pid_unb") for v in (0.0335, 0.067) for r in sel(g, v)]

for r in runs: r.update({"st_" + k: v for k, v in steadiness(r).items()})
rows_st = ""
for v in SP:
    for g in ["nopid", "nopid_unb", "pid", "pid_unb"]:
        rs = sel(g, v); a = lambda k: np.array([r["st_" + k] for r in rs])
        hl = " class=hl" if g.endswith("_unb") else ""
        rows_st += (f"<tr{hl}><td>{GN[g]}</td><td>{v}</td><td>{a('drift').mean():+.2f} ± {a('drift').std(ddof=1):.2f}</td>"
                    f"<td>{a('drift').min():+.1f} ถึง {a('drift').max():+.1f}</td><td><b>{a('res_sd').mean():.2f}</b></td><td>{a('res_sd').max():.2f}</td><td>{a('gz_sd').mean():.2f}</td></tr>")
def stm(g, v, k): return np.mean([r["st_" + k] for r in sel(g, v)])
rows_heading = ""
for g, v in (("nopid", 0.134), ("nopid_unb", 0.134), ("pid", 0.134), ("pid_unb", 0.134)):
    rs = sel(g, v); ek = [r["yaw_ekf_end"] for r in rs] if g.startswith("pid") else None
    rows_heading += (f"<tr><td>{GN[g]}</td><td>{v}</td><td><b>{sg(M(rs, 'yaw_imu_end'))}</b></td><td>{sg(M(rs, 'yaw_wheel_end'))}</td>"
                     f"<td>{sg(np.mean(ek)) if ek else '–'}</td><td>{max(r['cmd_wz_absmax'] for r in rs):.3f}</td>"
                     f"<td>{np.mean([abs(r['lateral_cm']) for r in rs]):.1f}</td></tr>")

rows_runs = ""
for g in ORDER:
    for r in sel(g):
        flag = []
        if r["late_start"]: flag.append("ช่วงออกตัวไม่ครบ")
        if np.nanmax(r["pose_v"][1]) > 3 * r["v_set"]: flag.append("odom reset กลาง bag")
        if r["odom_gap"] > 0.9: flag.append(f"odom ขาด {r['odom_gap']:.1f} s")
        rows_runs += (f"<tr><td class=mono>{html.escape(r['name'])}</td><td>{GN[r['group']]}</td><td>{r['v_set']}</td><td>{f(r['cmd_mean'])}</td>"
                      f"<td>{f(r['odom_v_pose'])}</td><td>{sg(r['err_pct'],1)}%</td>"
                      f"<td>{r['yaw_sd']:.2f}</td><td>{r['yaw_absmax']:.2f}</td><td>{sg(r['yaw_imu_end'])}</td><td>{sg(r['yaw_wheel_end'])}</td><td>{', '.join(flag)}</td></tr>")

doc = f"""<!DOCTYPE html><html lang="th"><head><meta charset="utf-8"><title>TurtleBot3: NO PID vs PID</title>
<style>
@page {{ size: A4; margin: 16mm 14mm 16mm 14mm; }}
body {{ font-family: 'Noto Sans Thai', 'Noto Sans', sans-serif; font-size: 10.5pt; line-height: 1.6; color: #2b2b29; }}
h1 {{ font-size: 18pt; margin: 0 0 2px; }} .sub {{ color: #6b6a65; margin-bottom: 14px; }}
h2 {{ font-size: 13.5pt; border-bottom: 1.5px solid #d9d8d3; padding-bottom: 3px; margin-top: 22px; }}
h3 {{ font-size: 11.5pt; margin: 14px 0 4px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 6px 0 4px; font-variant-numeric: tabular-nums; }}
th, td {{ border-bottom: 1px solid #e6e5e1; padding: 3px 5px; text-align: right; vertical-align: top; }}
th {{ background: #f3f2ee; text-align: right; font-weight: 600; }} td:first-child, th:first-child {{ text-align: left; }}
tr.hl td {{ background: #fdf3e1; }} td {{ white-space: nowrap; }} .runs td:last-child {{ white-space: normal; min-width: 90px; }}
.runs {{ font-size: 7.8pt; }} .runs td:nth-child(2), .runs td:last-child {{ text-align: left; }}
.m {{ color: #6b6a65; }} .mono {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 7.2pt; }}
.box {{ background: #f6f5f1; border-left: 4px solid #2a78d6; padding: 8px 12px; margin: 8px 0; }}
.box li {{ margin: 3px 0; }} figure {{ margin: 8px 0 12px; break-inside: avoid; }} figure svg {{ width: 100%; height: auto; }}
figcaption {{ font-size: 9pt; color: #5a5a57; }} .note {{ font-size: 9pt; color: #5a5a57; }}
.pb {{ break-before: page; }} h2, h3 {{ break-after: avoid; }} td:first-child, .runs td:nth-child(2) {{ white-space: nowrap; }} table, .box {{ break-inside: avoid; }}
</style></head><body>
<h1>TurtleBot3: เปรียบเทียบ NO PID กับ PID การวิ่งตรง 1.2 m</h1>
<div class=sub>ความเร็วที่สั่ง (cmd_vel) · ความเร็วที่รถวิ่งได้จริง · ความนิ่งของมุม yaw · รวมกรณีถ่วงมวลไม่สมดุล (unbalance) &nbsp;|&nbsp; rosbag {len(runs)} run</div>

<div class=box><b>สรุปผล</b><ul>
<li><b>NO PID วิ่งช้ากว่าที่สั่ง</b> ที่ 0.0335 และ 0.067 m/s ช้ากว่า <b>~12%</b> ที่ 0.134 m/s ช้ากว่า ~3% ค่านี้คงที่มากในทุก run จึงเป็น error เชิงระบบ (bias) ไม่ใช่ noise</li>
<li><b>PID ชดเชยได้</b> โดยส่ง cmd สูงกว่า setpoint 5–18% ทำให้ความเร็วจริงอยู่ห่าง setpoint แค่ <b>+1.1 ถึง +1.4%</b> ทุกความเร็ว</li>
<li><b>การจูนช่วยเรื่องช่วงออกตัว</b> ก่อนจูน cmd พุ่งไป <b>2.1–2.2 เท่า</b>ของ setpoint หลังจูนเหลือ 1.2–1.4 เท่า ข้อเสียคือ cmd ถูกปัดเป็นขั้นละ 0.01 m/s และสลับไปมาตลอด</li>
<li><b>หุ่นสมดุล (ไม่ถ่วงมวล) วิ่งตรงทุกกลุ่ม:</b> yaw เบี่ยงไม่เกิน {ymax_bal:.1f}° ({n15_bal}/{len(bal)} run ไม่เกิน 1.5°)</li>
<li><b>Unbalance (ถ่วง 358 g ด้านซ้าย):</b> ที่ 0.0335 และ 0.067 m/s ยังวิ่งตรงเหมือนเดิม แต่ <b>ที่ 0.134 m/s NO PID หันขวาสะสม {sg(nu_end,1)}°</b> (ทั้ง 5 run, ปลายทางเยื้องขวา ~{abs(nu_lat):.0f} cm) PID ลดเหลือเฉลี่ย {M(pu, 'yaw_imu_end'):+.1f}° แต่แต่ละ run ต่างกันมาก (−7.2 ถึง +2.5°) จึงยังแก้ทิศได้ไม่หมด</li>
<li><b>ความนิ่งเมื่อ unbalance:</b> NO PID เลี้ยวอย่างนิ่ง ๆ (ส่ายแค่ ~0.2° เท่าหุ่นสมดุล) ส่วน PID ที่ 0.134 m/s yaw ส่ายไปมา ±2–3° (SD หลังหัก drift 0.92° ประมาณ 3 เท่าของหุ่นสมดุล) PID จึงลดการเลี้ยวได้แต่แลกกับความนิ่ง</li>
<li><b>สาเหตุ:</b> ล้อไถล encoder ล้อจึงมองไม่เห็นการหมุนนี้ (ล้อสองข้างหมุนเท่ากันแต่ตัวรถหมุน) และ yaw ของ /odom_ekf ก็แทบไม่เปลี่ยน (~{M(pu, 'yaw_ekf_end'):+.1f}°) มีแค่ IMU ที่วัดได้</li>
</ul></div>

<h2>1. ข้อมูลและวิธีวัด</h2>
<p>รูปแบบชื่อโฟลเดอร์: <span class=mono>(nopid|pid)_ระยะ_ความเร็วที่สั่ง_(ระยะที่วัดได้ | ครั้งที่ | unbalance_ครั้งที่)</span> ทุกชุดวิ่ง 1.2 m แบ่งเป็น 5 กลุ่ม กลุ่มละ 5 run ต่อความเร็ว</p>
<ul>
<li><b>NO PID</b> และ <b>PID (หลังจูน)</b> ที่ 0.0335 / 0.067 / 0.134 m/s (กลุ่มละ 15 run)</li>
<li><b>NO PID unbalance</b> และ <b>PID unbalance</b> ที่ 0.0335 / 0.067 / 0.134 m/s (กลุ่มละ 15 run) ถ่วงมวล <b>358 g ติดด้านซ้าย</b>ของหุ่น</li>
<li><b>PID (ก่อนจูน)</b> จากโฟลเดอร์ <span class=mono>ZBefore_Tune</span> ที่ 0.0335 / 0.067 m/s (10 run)</li>
</ul>
<p><b>ความเร็ว cmd</b> = ค่าเฉลี่ย <span class=mono>cmd_vel.linear.x</span> ในช่วงวิ่งนิ่ง (ตั้งแต่ 1.5 s หลังออกตัว ถึง 0.5 s ก่อนหยุด)<br>
<b>ความเร็วจริง</b> = ระยะทางที่ได้จากตำแหน่ง <span class=mono>/odom</span> (pose) หารด้วยเวลา ในช่วงเดียวกัน ค่านี้ตรงกับระยะที่วัดด้วยตลับเมตร ส่วนค่า <span class=mono>/odom</span> twist สูงกว่าความจริงราว 2–3% จึงไม่ใช้<br>
<b>Yaw</b> = มุมจาก <span class=mono>/imu</span> เทียบกับตอนก่อนออกตัว (+ = หันซ้าย/CCW) ตอนจอดนิ่ง IMU drift มีค่ามัธยฐานแค่ ~0.03°/นาที<br>
<b>Yaw จากล้อ</b> = r·(θ<sub>R</sub>−θ<sub>L</sub>)/L จาก <span class=mono>/joint_states</span> (Burger: r = 0.033 m, L = 0.160 m) ใช้ดูว่าล้อเลี้ยวเท่าไร เทียบกับตัวรถที่หมุนจริง (IMU)</p>
<p class=note>หมายเหตุ: yaw ใน <span class=mono>/odom</span> เท่ากับ IMU ทุก run (odom เอามุมมาจาก IMU) run ที่ช่วงออกตัวไม่ครบ (เริ่มอัด bag หรือ odom เริ่มบันทึกหลังรถออกตัวแล้ว) ไม่นำไปคิด cmd peak และเวลาวิ่ง ดูรายละเอียดในภาคผนวก</p>

<h2>2. ความเร็ว: cmd_vel เทียบกับความเร็วจริง</h2>
<table><tr><th>กลุ่ม</th><th>setpoint<br>[m/s]</th><th>n</th><th>cmd เฉลี่ย [m/s]<br>(เทียบ setpoint)</th><th>ความเร็วจริง [m/s]<br>เฉลี่ย ± SD</th><th>error<br>จริง</th><th>/odom_ekf<br>[m/s]</th><th>cmd peak<br>/ setpoint</th><th>cmd แกว่ง<br>(SD/setpoint)</th><th>เวลาวิ่ง<br>1.2 m [s]</th></tr>
{rows_speed}</table>
<p class=note>เวลาวิ่งในอุดมคติ (1.2 m / setpoint) = 35.8 / 17.9 / 9.0 s</p>
<figure>{svg("f1_speed.svg")}<figcaption>รูปที่ 1 ซ้าย: cmd_vel ที่ส่งจริงตอนวิ่งนิ่ง ขวา: ความเร็วจริง ทั้งสองเทียบกับ setpoint เป็น % จุด = แต่ละ run, ขีดดำ = ค่าเฉลี่ย</figcaption></figure>

<h3>วิเคราะห์</h3>
<ul>
<li><b>NO PID:</b> ได้ความเร็วจริงแค่ 87–89% ของที่สั่งที่ความเร็วต่ำ และ ~97% ที่ 0.134 m/s แต่ละ run ซ้ำกันเกือบเป๊ะ จึงเป็น gain error ของระบบขับ ทำให้ใช้เวลาวิ่ง 1.2 m นานกว่าที่ควรประมาณ 15%</li>
<li><b>PID:</b> ระดับการชดเชย (cmd สูงกว่า setpoint +18% / +9% / +5%) ตรงกับขนาด error ของ NO PID ความเร็วจริงจึงเหลือ error แค่ ~+1%</li>
<li><b>ก่อนจูนกับหลังจูน:</b> ความเร็วตอนวิ่งนิ่งต่างกันไม่เกิน 0.6% ส่วนที่ต่างชัดคือช่วงออกตัว (รูปที่ 3)</li>
<li><b>Unbalance กับความเร็ว:</b> ที่ 0.0335 และ 0.067 m/s ความเร็วแทบไม่เปลี่ยน ที่ 0.134 m/s NO PID unbalance ช้าลงเป็น −4.6% (จาก −2.9%) และ PID unbalance เร็วเกินเป็น +3.3% (จาก +1.1%) ทั้งสองกลุ่มกระจายระหว่าง run มากขึ้น ซึ่งไปด้วยกันกับล้อที่ไถลที่ความเร็วนี้ (หัวข้อ 3.2–3.4)</li>
</ul>

<figure>{svg("f2_timeseries.svg")}<figcaption>รูปที่ 2a ความเร็วตามเวลา หุ่นสมดุล (run ที่ใกล้ค่ามัธยฐาน) เส้นเทา = cmd_vel, เส้นสี = ความเร็วจริงจากตำแหน่ง odom (หน้าต่าง 0.5 s), เส้นประ = /odom_ekf, เส้นจุด = setpoint จุดพุ่ง/ตกเป็นครั้งคราวมาจาก timestamp ของ odom ที่ไม่สม่ำเสมอ</figcaption></figure>
<figure>{svg("f2c_timeseries_unb.svg")}<figcaption>รูปที่ 2b ความเร็วตามเวลา กรณี unbalance (รูปแบบเดียวกับรูปที่ 2a)</figcaption></figure>

<figure>{svg("f2b_kick.svg")}<figcaption>รูปที่ 3 cmd_vel / setpoint ใน 3 วินาทีแรก (ทุก run ที่ช่วงออกตัวครบ) ก่อนจูน (เขียว) cmd พุ่ง 2–2.8 เท่า หลังจูน (ส้ม) และ PID unbalance (ชมพู) เหลือ ~1.2–1.6 เท่า</figcaption></figure>
<ul>
<li><b>cmd ของ PID หลังจูนเป็นขั้นบันได:</b> ค่า cmd ช่วงวิ่งนิ่งมีแค่ 2–3 ค่า ห่างกันทีละ 0.01 m/s (เช่น 0.0335 m/s: 0.035 ↔ 0.045) ค่าเฉลี่ยจึงตรง แต่ล้อถูกสั่งสลับความเร็วตลอด ที่ 0.0335 m/s ขั้นละ 0.01 คิดเป็น ~30% ของ setpoint ส่วน PID ก่อนจูนให้ค่าต่อเนื่อง</li>
</ul>

<h2 class=pb>3. ความนิ่งของมุม Yaw (หุ่นวิ่งตรงไหม)</h2>
<p>ถ้าหุ่นวิ่งตรง yaw ควรอยู่ใกล้ 0° (ทิศตอนเริ่ม) ตลอดทาง ส่วนนี้ดูว่า yaw ขยับไปจากทิศเริ่มต้นมากแค่ไหนในช่วงที่รถวิ่ง (ตั้งแต่ออกตัวจนหยุด) แถวที่ไฮไลต์คือกลุ่มที่มี run เบี่ยงเกิน 3°</p>
<table><tr><th>กลุ่ม</th><th>setpoint<br>[m/s]</th><th>n</th><th>SD ของ yaw<br>ระหว่างวิ่ง [°]</th><th>ช่วงแกว่ง<br>(max−min) [°]</th><th>|yaw| สูงสุด<br>เฉลี่ย [°]</th><th>|yaw| สูงสุด<br>แย่ที่สุด [°]</th><th>yaw ตอนจบ [°]<br>เฉลี่ย ± SD</th></tr>
{rows_yaw}</table>
<figure>{svg("f3_yaw.svg")}<figcaption>รูปที่ 4a yaw (IMU) เทียบกับระยะที่วิ่ง หุ่นสมดุล แต่ละเส้นคือหนึ่ง run แถว = ความเร็ว, คอลัมน์ = กลุ่ม</figcaption></figure>
<figure>{svg("f3b_yaw_unb.svg")}<figcaption>รูปที่ 4b yaw (IMU) เทียบกับระยะที่วิ่ง กรณี unbalance (สเกลแกน y กว้างกว่ารูปที่ 4a)</figcaption></figure>
<figure>{svg("f4_yaw_end.svg")}<figcaption>รูปที่ 5 ซ้าย: |yaw| สูงสุดระหว่างวิ่ง ขวา: ช่วงแกว่งของ yaw (max−min) จุด = แต่ละ run, ขีดดำ = ค่าเฉลี่ย</figcaption></figure>

<h3>3.1 หุ่นสมดุล</h3>
<ul>
<li><b>yaw นิ่ง:</b> ระหว่างวิ่ง yaw ขยับในช่วงแคบ ๆ ราว 1–1.5° (max−min) และ SD แค่ 0.2–0.4° ไม่มีการส่ายใหญ่ ทั้งสามกลุ่มและทุกความเร็วนิ่งพอ ๆ กัน</li>
<li><b>เบี่ยงน้อย:</b> |yaw| สูงสุดเฉลี่ย 0.7–1.1° มีแค่ {len(bal) - n15_bal} run ที่เกิน 1.5° (สูงสุด {ymax_bal:.1f}°) ทุกตัวเป็น NO PID ที่ yaw ค่อย ๆ ไหลไปทางซ้าย ({left_np}/15 run) ปลายทางเยื้องด้านข้างไม่เกิน ~2.5 cm (มัธยฐาน 0.5 cm, จากตำแหน่ง odom) <b>สรุปว่าวิ่งตรง</b></li>
</ul>
<h3>3.2 Unbalance (ถ่วง 358 g ด้านซ้าย)</h3>
<ul>
<li><b>ความเร็วต่ำยังวิ่งตรง:</b> ที่ 0.0335 และ 0.067 m/s ส่วนใหญ่ yaw อยู่ในระดับเดียวกับหุ่นสมดุล (|yaw| สูงสุดเฉลี่ย 1.0–1.6°) แต่ PID unbalance ที่ 0.0335 m/s มี 1 run เบี่ยงถึง {max(lowspd_unb):.1f}°</li>
<li><b>ที่ 0.134 m/s NO PID หันขวาชัดเจน:</b> ทุก run yaw ลดลงเป็นเส้นตรงประมาณ {nu_rate:.1f}°/s ตั้งแต่ออกตัวจนหยุด จบที่ {sg(nu_end,1)} ± {np.std([r['yaw_imu_end'] for r in nu], ddof=1):.1f}° ปลายทางเยื้องขวาประมาณ {abs(nu_lat):.0f} cm (จากตำแหน่ง odom) <b>ไม่ได้วิ่งตรง</b></li>
<li><b>PID ช่วยได้บางส่วน:</b> ที่ 0.134 m/s yaw ตอนจบเฉลี่ย {sg(M(pu, 'yaw_imu_end'),1)}° แต่กระจายมาก (−7.2 ถึง +2.5°) |yaw| สูงสุดระหว่างวิ่ง 2.7–7.7° และ SD ของ yaw ระหว่างวิ่งสูงกว่าหุ่นสมดุล 3–4 เท่า ปลายทางเยื้องเฉลี่ย ~{np.mean(pu_lat):.0f} cm</li>
</ul>

<h3>3.3 ความนิ่งของ yaw เมื่อ unbalance: แยก "ไหลไปทางเดียว" กับ "ส่ายไปมา"</h3>
<p>SD ของ yaw ในตารางข้างบนรวมสองเรื่องไว้ด้วยกัน คือ (1) <b>drift</b> = yaw ไหลไปทางเดียวสะสมตามระยะ (ความชันของเส้นตรงที่ fit yaw เทียบกับระยะ หน่วย °/m) และ (2) <b>การส่าย</b> = SD ของ yaw หลังหักเส้นแนวโน้มนั้นออก ถ้าหุ่นนิ่งค่านี้ต้องต่ำ ใช้ช่วงตั้งแต่ 1 s หลังออกตัวถึง 0.5 s ก่อนหยุด และดู SD ของอัตราหมุน (gyro z) ประกอบ</p>
<table><tr><th>กลุ่ม</th><th>setpoint<br>[m/s]</th><th>drift [°/m]<br>เฉลี่ย ± SD</th><th>drift<br>ช่วง (5 run)</th><th>การส่าย: SD หลังหัก<br>drift เฉลี่ย [°]</th><th>การส่าย<br>แย่ที่สุด [°]</th><th>SD ของ gyro z<br>[°/s]</th></tr>
{rows_st}</table>
<figure>{svg("f6_yaw_steadiness.svg")}<figcaption>รูปที่ 6 ซ้าย: drift ของ yaw (°/m, + = ไหลไปทางซ้าย) ขวา: การส่ายของ yaw หลังหัก drift ออก จุด = แต่ละ run, ขีดดำ = ค่าเฉลี่ย</figcaption></figure>
<figure>{svg("f7_yaw_detrended.svg")}<figcaption>รูปที่ 7 yaw หลังหัก drift ออก ที่ 0.134 m/s (5 run ต่อกลุ่ม) ดูว่าหุ่นส่ายแค่ไหน</figcaption></figure>
<ul>
<li><b>NO PID unbalance = เลี้ยวอย่างนิ่ง ๆ ไม่ได้ส่าย:</b> ที่ 0.134 m/s yaw ไหลไปทางขวา {stm('nopid_unb',0.134,'drift'):+.1f} °/m แทบเท่ากันทุก run แต่การส่ายหลังหัก drift แค่ {stm('nopid_unb',0.134,'res_sd'):.2f}° ใกล้หุ่นสมดุล ({stm('nopid',0.134,'res_sd'):.2f}°) ที่ความเร็วต่ำ ทั้ง drift และการส่ายใกล้เคียงหุ่นสมดุล (การส่าย 0.16–0.27°) <b>ปัญหาของ NO PID unbalance จึงเป็นเรื่องทิศไหล ไม่ใช่ความนิ่ง</b></li>
<li><b>PID unbalance = ส่ายมากที่สุด:</b> ที่ 0.134 m/s การส่ายเฉลี่ย <b>{stm('pid_unb',0.134,'res_sd'):.2f}°</b> (สูงสุด 1.6°) ประมาณ 3 เท่าของ PID หุ่นสมดุล ({stm('pid',0.134,'res_sd'):.2f}°) และ gyro z แกว่ง {stm('pid_unb',0.134,'gz_sd'):.1f}°/s (PID หุ่นสมดุล {stm('pid',0.134,'gz_sd'):.1f}°/s) รูปที่ 7 แสดงว่า yaw แกว่งช้า ๆ ±2–3° รอบละประมาณ 0.5 m (~3–4 s) ลักษณะเหมือนตัวควบคุมแก้ทิศเกินแล้วต้องแก้กลับ drift ก็ไม่สม่ำเสมอ (−4.2 ถึง +2.4 °/m)</li>
<li><b>PID unbalance ที่ความเร็วต่ำ:</b> การส่ายยังสูงกว่าหุ่นสมดุลราว 2 เท่า (0.34° และ 0.27° เทียบกับ 0.17° และ 0.15°) และมี 1 run ที่ 0.0335 m/s ไหลไปทางซ้าย +4.1 °/m แต่ขนาดยังเล็ก (เบี่ยงไม่เกิน ~1.5° ยกเว้น run นั้น)</li>
<li><b>สรุปความนิ่ง:</b> เมื่อไม่สมดุล NO PID ยังนิ่งแต่เลี้ยว ส่วน PID ลดการเลี้ยวได้แต่แลกกับการส่ายที่มากขึ้น สาเหตุน่าจะมาจาก feedback ทิศที่ไม่เห็นการไถล (หัวข้อ 3.4)</li>
</ul>

<h3>3.4 ทำไม PID แก้ทิศได้ไม่หมด: เทียบมุมจาก 3 แหล่ง</h3>
<table><tr><th>กลุ่ม</th><th>setpoint</th><th>yaw IMU<br>(ตัวรถหมุนจริง) [°]</th><th>yaw จากล้อ<br>(encoder) [°]</th><th>yaw /odom_ekf<br>[°]</th><th>|angular.z| cmd<br>สูงสุด [rad/s]</th><th>เยื้องด้านข้าง<br>เฉลี่ย [cm]</th></tr>
{rows_heading}</table>
<p class=note>ค่าทั้งหมดคือมุมตอนจบเทียบกับตอนเริ่ม เฉลี่ย 5 run</p>
<figure>{svg("f5_heading_sources.svg")}<figcaption>รูปที่ 8 yaw ตามเวลาที่ 0.134 m/s กรณี unbalance เส้นทึบ = IMU (ตัวรถหมุนจริง), เส้นประเทา = มุมที่ล้อเลี้ยว (encoder), เส้นจุด = /odom_ekf</figcaption></figure>
<ul>
<li><b>NO PID unbalance:</b> ล้อสองข้างหมุนเท่ากัน (yaw จากล้อ ≈ 0°) แต่ตัวรถหมุนไปทางขวา ~10° แปลว่า<b>ล้อไถล</b> encoder จึงมองไม่เห็น สาเหตุที่น่าจะเป็น: มวลด้านซ้ายทำให้ล้อขวารับน้ำหนักน้อยลง ล้อขวาจึงยึดเกาะน้อยและไถล ด้านขวาจึงเคลื่อนช้ากว่าด้านซ้าย รถเลยหันขวา (ยังไม่ได้ยืนยันด้วยการทดลอง)</li>
<li><b>PID unbalance:</b> PID สั่ง angular.z สูงถึง {max(r['cmd_wz_absmax'] for r in pu):.3f} rad/s (หุ่นสมดุลสูงสุด 0.065) ล้อจึงเลี้ยวซ้ายรวม ~{M(pu, 'yaw_wheel_end'):+.0f}° เพื่อแก้ แต่ตัวรถจริงยังอยู่ที่ {sg(M(pu, 'yaw_imu_end'),1)}° ส่วน yaw ของ /odom_ekf อยู่ที่ ~{sg(M(pu, 'yaw_ekf_end'),1)}° ตลอด ถ้า PID ใช้ EKF yaw เป็น feedback (รูปที่ 8 บ่งชี้แบบนั้น เพราะ EKF ถูกคุมไว้ที่ ~0) ตัวควบคุมจะคิดว่ารถตรงแล้วทั้งที่ตัวรถยังเบี่ยง <b>EKF เชื่อข้อมูลล้อมากกว่า IMU จึงไม่เห็นการไถล</b></li>
</ul>

<h2>4. ข้อสังเกตและข้อเสนอแนะ</h2>
<ul>
<li>Error ความเร็วของ NO PID เป็น bias ที่ซ้ำได้ ถ้าไม่ใช้ PID สามารถชดเชยด้วย feed-forward gain ตามความเร็วได้ (×1.14 ที่ 0.0335, ×1.13 ที่ 0.067, ×1.03 ที่ 0.134 m/s)</li>
<li><b>เรื่องแก้ทิศเมื่อไม่สมดุล:</b> ให้ feedback ทิศของ PID ใช้ yaw จาก IMU โดยตรง หรือปรับ EKF ให้เชื่อ IMU (gyro z / yaw) มากขึ้นและเชื่อ yaw จากล้อน้อยลง (เพิ่ม covariance ของ yaw จาก wheel odom) เพื่อให้เห็นการไถล</li>
<li>ลดการไถลที่ 0.134 m/s เช่น ใส่ ramp เร่งความเร็วให้ช้าลง หรือจัดมวลให้สมดุล/ต่ำลง แล้วทดลองซ้ำเพื่อยืนยันสาเหตุ (เช่นย้ายมวลไปด้านขวาแล้วดูว่าหันซ้ายแทนไหม)</li>
<li>ควรแก้การปัดค่า output ของ PID ทีละ 0.01 m/s ให้ละเอียดขึ้น หรือใส่ filter ที่ output เพื่อลดการสลับ cmd ตลอดเวลา</li>
<li>อัด bag ให้ครอบช่วงก่อนออกตัวทุกครั้ง และเช็กว่า odom ไม่ขาดหาย</li>
</ul>

<h2 class=pb>ภาคผนวก: ผลรายตัว</h2>
<table class=runs><tr><th>run</th><th>กลุ่ม</th><th>setpoint</th><th>cmd</th><th>v จริง</th><th>error</th><th>SD yaw [°]</th><th>|yaw| max [°]</th><th>yaw IMU จบ [°]</th><th>yaw ล้อ จบ [°]</th><th>หมายเหตุ</th></tr>
{rows_runs}</table>
</body></html>"""
open(R + "report.html", "w").write(doc)
print("html ok")
