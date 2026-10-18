"""Print a dance x condition table of completed % from day4_eval summary.txt files.
Usage: python eval_table.py <out_dir> <name>   (folders <name>__<clip>__<condition>)"""
import glob, os, sys

out, name = sys.argv[1], sys.argv[2]
order = ["none", "gains", "mass", "friction", "noise", "delay", "gains_mass", "all"]
rows = {}
for d in sorted(glob.glob(os.path.join(out, f"{name}__*__*"))):
    _, clip, cond = os.path.basename(d).split("__")
    f = os.path.join(d, "summary.txt")
    if not os.path.exists(f):
        continue
    kv = {}
    for line in open(f):
        p = line.split(None, 1)
        if len(p) == 2:
            kv[p[0]] = p[1].strip()
    rows.setdefault(clip, {})[cond] = kv
print(f"{name}: completed % (tracked s) per dance and condition; 100 = whole dance without falling")
print(f"{'dance':22s}" + "".join(f"{c:>13s}" for c in order))
for clip, conds in rows.items():
    cells = []
    for c in order:
        kv = conds.get(c)
        cells.append(f"{float(kv['completed_pct']):5.0f} ({float(kv['tracked_s']):4.1f})" if kv else "-")
    print(f"{clip:22s}" + "".join(f"{x:>13s}" for x in cells))
print("\njoint rmse [rad], clean run:")
for clip, conds in rows.items():
    kv = conds.get("none")
    if kv:
        print(f"  {clip:20s} {kv.get('joint_rmse_mean_rad', '-')}   worst: {kv.get('worst_joints_rmse', '-')[:90]}")
