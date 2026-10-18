"""Tables from robust_eval.py results. Usage: python robust_table.py <name> [<name> ...]
(reads ~/projects/g1_pilot/robust/<name>__<clip>__<test>/results.json; several names = side by side)"""
import glob, json, os, sys

root = os.path.expanduser("~/projects/g1_pilot/robust")
names = sys.argv[1:] or ["p2"]
R = {}  # (clip, test) -> name -> results
for n in names:
    for d in sorted(glob.glob(os.path.join(root, f"{n}__*__*"))):
        f = os.path.join(d, "results.json")
        if os.path.exists(f):
            _, clip, test = os.path.basename(d).split("__")
            R.setdefault((clip, test), {})[n] = json.load(open(f))
clips = sorted({c for c, _ in R})
w = max(10, max(len(n) for n in names) + 2)
print("Monte Carlo: % of robots that finish the whole dance (random dynamics each)")
print(f"{'dance':16s}{'test':12s}" + "".join(f"{n:>{w}s}" for n in names))
for c in clips:
    for t, label in (("mc", "training DR"), ("mc_w1.5", "1.5x wider")):
        if (c, t) in R:
            row = R[(c, t)]
            print(f"{c:16s}{label:12s}" + "".join(f"{(str(row[n]['success_pct']) + '%') if n in row else '-':>{w}s}" for n in names))
for c in clips:
    for f in ("mass", "gains", "friction", "delay"):
        row = R.get((c, f"sweep_{f}"))
        if not row:
            continue
        print(f"\nsweep {f} ({c}): success % per value, other factors random in training DR")
        vals = sorted({s["value"] for r in row.values() for s in r["sweep"]})
        print(f"{'value':>8s}" + "".join(f"{n:>{w}s}" for n in names))
        for v in vals:
            cells = []
            for n in names:
                hit = [s for s in row.get(n, {}).get("sweep", []) if s["value"] == v]
                cells.append(f"{hit[0]['success_pct']:.0f}%" if hit else "-")
            print(f"{v:8.2f}" + "".join(f"{x:>{w}s}" for x in cells))
for c in clips:
    row = R.get((c, "mc"), {})
    for n in names:
        r = row.get(n)
        if r and r.get("fall_seconds_top"):
            print(f"\n{n} {c} training-DR falls by time: {r['fall_seconds_top']}  causes: {r.get('causes')}")
