"""Re-run every terminated run of the n512 MuJoCo tests to the clip end (same seed / DR, no fall stop) and label it:
REAL FALL = pelvis below 0.45 m or torso tilt above 60 deg at any time; DRIFT = stayed upright but crossed a training
termination threshold. Writes sim2sim/results/fall_classes_n512.json and prints a table.

conda run -n gmr python sim2sim/classify_falls.py
"""
import importlib.util
import json
import math
import os
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.expanduser(os.environ.get("G1_DANCE_DATA", "~/projects/g1_pilot"))  # day1/, day2/ motion files
spec = importlib.util.spec_from_file_location("s2s", os.path.join(HERE, "sim2sim_dance.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["s2s"] = mod
spec.loader.exec_module(mod)

RES = os.path.join(HERE, "results")
L = os.path.expanduser("~/projects/unitree_rl_lab/logs/rsl_rl/g1_schuhplattler_residual/")
CK = dict(floor=L + "2026-10-31_22-40-36_sch_v3s_floor/model_34998.pt",
          plus=L + "2026-10-30_20-26-10_sch_v3s_plus/model_29999.pt")
TESTS = dict(flat=dict(widen=1.0, friction=None), w15=dict(widen=1.5, friction=None),
             f025=dict(widen=1.0, friction=0.25), f03=dict(widen=1.0, friction=0.3))
BASE = dict(motion=os.path.join(DATA, "day2", "schuhplattler_gmr_v3s.npz"), action="mimic",
            xml="~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml", start=0.0, end=None, sim_dt=0.005,
            viewer=False, speed=1.0, no_wait=False, close_at_end=False, record=None, width=1280, height=720,
            cam_azimuth=0.0, cam_elevation=-10.0, cam_distance=3.5, no_fall_stop=True, runs=0, kp_scale=None,
            mass_scale=None, terrain="flat", terrain_seed=None, delay_substeps=None, procs=1, seed=0, name=None, out="")
FALL_Z, FALL_TILT, THR = 0.45, 60.0, 0.25


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def replay(job):
    pol, test, seed, t_term, cause = job
    r = mod.run_once(dict(args=dict(BASE, checkpoint=CK[pol], **TESTS[test]), seed=seed, random=True))
    fell = r["min_pelvis_z"] < FALL_Z or r["max_torso_tilt_deg"] > FALL_TILT
    return dict(policy=pol, test=test, seed=seed, t_term=t_term, cause=cause, real_fall=bool(fell),
                t_fall_first_term=t_term, **{k: r[k] for k in ("min_pelvis_z", "max_torso_tilt_deg", "max_ee_err",
                                                            "max_anchor_err", "end_ee_err", "end_anchor_err", "body_err")})


if __name__ == "__main__":
    jobs, n_runs = [], {}
    for pol in CK:
        for test in TESTS:
            runs = json.load(open(os.path.join(RES, f"v3s_{pol}_{test}_n512.json")))["runs"]
            n_runs[(pol, test)] = len(runs)
            jobs += [(pol, test, r["seed"], r["t_fail"], r["cause"]) for r in runs if not r["success"]]
    print(f"replaying {len(jobs)} terminated runs")
    with Pool(int(os.environ.get("PROCS", 20))) as p:
        out = p.map(replay, jobs)
    json.dump(out, open(os.path.join(RES, "fall_classes_n512.json"), "w"), indent=1)
    print(f"{'test':6s} {'policy':6s} {'n':>4s} {'upright%':>9s} {'real fall%  [95% CI]':>22s} {'drift%':>7s}  real-fall times [s]")
    for test in TESTS:
        for pol in CK:
            n = n_runs[(pol, test)]
            o = [x for x in out if x["policy"] == pol and x["test"] == test]
            fall = [x for x in o if x["real_fall"]]
            drift = [x for x in o if not x["real_fall"]]
            lo, hi = wilson(len(fall), n)
            ts = sorted(x["t_term"] for x in fall)
            print(f"{test:6s} {pol:6s} {n:4d} {100 * (n - len(fall)) / n:9.1f} {100 * len(fall) / n:8.1f} [{lo:4.1f}-{hi:4.1f}] "
                  f"{100 * len(drift) / n:7.1f}  {ts[:8]}{' ...' if len(ts) > 8 else ''}")
            if drift:
                rec = sum(x["end_ee_err"] < THR and x["end_anchor_err"] < THR for x in drift)
                mx = sorted(x["max_ee_err"] for x in drift)
                print(f"{'':14s} drift: max ee err median {mx[len(mx) // 2]:.2f} m (max {mx[-1]:.2f}), "
                      f"back under 0.25 m at the end {rec}/{len(drift)}, end body err median "
                      f"{100 * sorted(x['body_err'] for x in drift)[len(drift) // 2]:.1f} cm")
