"""Print the sim2sim results folder as one table: python table.py <results dir>"""
import glob
import json
import os
import sys

import numpy as np

rows = []
for f in sorted(glob.glob(os.path.join(os.path.expanduser(sys.argv[1]), "*.json"))):
    d = json.load(open(f))
    n, runs = d["nominal"], d["runs"]
    ok = [r for r in runs if r["success"]]
    tf = [r["t_fail"] for r in runs if not r["success"]]
    rows.append((os.path.basename(f)[:-5],
                 "OK" if n["success"] else f"fell {n['t_fail']:.1f}s",
                 f"{100 * len(ok) / len(runs):.0f}% ({len(ok)}/{len(runs)})" if runs else "-",
                 f"{np.median(tf):.1f}" if tf else "-",
                 f"{np.mean([r['joint_rmse'] for r in ok]):.3f}" if ok else "-",
                 f"{100 * np.mean([r['body_err'] for r in ok]):.1f}" if ok else "-"))
head = ("policy", "nominal", "MC finished", "median fall [s]", "joint rmse", "body err [cm]")
w = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
for r in [head] + rows:
    print("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
