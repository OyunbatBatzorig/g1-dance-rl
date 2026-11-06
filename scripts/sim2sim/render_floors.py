"""Render the DRFloor-SchV3s floor tiles (same generators as apply_floor_mix) with the G1 standing on each.

MUJOCO_GL=egl python sim2sim/render_floors.py --out media/floor_mix.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

H_SCALE, V_SCALE = 0.1, 0.005  # TerrainGeneratorCfg horizontal_scale / vertical_scale
CROP = 3.0  # rendered patch [m] around the tile centre (robot spawn point)


def random_uniform(rng, n, noise_max, step=0.005):
    # Isaac hf random_uniform_terrain with downsampled_scale == horizontal_scale: one random height per 10 cm cell
    levels = np.arange(0, int(noise_max / V_SCALE) + int(step / V_SCALE), int(step / V_SCALE))
    return rng.choice(levels, size=(n, n)) * V_SCALE


def pyramid_slope(n_tile, slope, platform_width):
    # Isaac hf pyramid_sloped_terrain on the full 8 m tile, then cropped around the centre
    height_max = int(slope * 8.0 / 2 / V_SCALE)
    c = n_tile // 2
    x = np.arange(n_tile)
    xx = ((c - np.abs(c - x)) / c).reshape(-1, 1)
    yy = ((c - np.abs(c - x)) / c).reshape(1, -1)
    hf = height_max * xx * yy
    pw = int(platform_width / H_SCALE / 2)
    z_pf = hf[c - pw, c - pw]
    hf = np.clip(hf, min(0, z_pf), max(0, z_pf))
    return np.rint(hf) * V_SCALE


def make_floors(seed):
    rng = np.random.default_rng(seed)
    n = int(CROP / H_SCALE)
    n_tile = int(8.0 / H_SCALE)
    full = pyramid_slope(n_tile, 0.05, 0.5)
    c = n_tile // 2
    slope = full[c - n // 2:c + n // 2, c - n // 2:c + n // 2]
    return [
        ("Flat (30%)", np.zeros((n, n))),
        ("Rough 0-1 cm (30%)", random_uniform(rng, n, 0.01)),
        ("Stone 0-2.5 cm (20%)", random_uniform(rng, n, 0.025)),
        ("Slope <=3 deg, 0.5 m top (20%)", slope - slope.min()),
    ]


def render(xml, hf, w, h):
    spec = mujoco.MjSpec.from_file(os.path.expanduser(xml))
    for g in [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]:  # the xml's own floor
        spec.delete(g)
    rng_z = max(float(hf.max() - hf.min()), 1e-4)
    field = spec.add_hfield(name="floor", size=[CROP / 2, CROP / 2, rng_z, 0.02],
                            nrow=hf.shape[0], ncol=hf.shape[1],
                            userdata=((hf - hf.min()) / rng_z).flatten().tolist())
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.62, 0.52, 0.40], rgb2=[0.55, 0.45, 0.34], width=300, height=300)
    mat = spec.add_material(name="floor_mat", texrepeat=[6, 6])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name, material="floor_mat",
                            pos=[0, 0, float(hf.min())])
    spec.worldbody.add_light(pos=[1.5, -1.5, 3.0], dir=[-0.5, 0.5, -1.0], diffuse=[0.8, 0.8, 0.8], castshadow=True)
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = w, h
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # lift the robot so its lowest collision geom rests on the floor under it
    feet = [g for g in range(model.ngeom) if model.geom_bodyid[g] > 0 and (model.geom_contype[g] or model.geom_conaffinity[g])]
    low = min(data.geom_xpos[g][2] - model.geom_rbound[g] for g in feet)
    i, j = hf.shape[0] // 2, hf.shape[1] // 2
    under = float(hf[i - 2:i + 3, j - 2:j + 3].max())
    data.qpos[2] += under - low
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.45 + under]
    cam.distance, cam.azimuth, cam.elevation = 3.2, 135, -22
    with mujoco.Renderer(model, h, w) as r:
        r.update_scene(data, cam)
        return r.render()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="~/projects/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")
    p.add_argument("--out", default="media/floor_mix.png")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    floors = make_floors(a.seed)
    fig, ax = plt.subplots(2, 4, figsize=(20, 9.5), gridspec_kw={"height_ratios": [1.15, 1]})
    for k, (name, hf) in enumerate(floors):
        ax[0, k].imshow(render(a.xml, hf, 640, 560))
        ax[0, k].set_title(name, fontsize=14)
        ax[0, k].axis("off")
        im = ax[1, k].imshow((hf - hf.min()) * 100, cmap="terrain", origin="lower",
                             extent=[-CROP / 2, CROP / 2, -CROP / 2, CROP / 2], vmin=0, vmax=max(2.5, np.ptp(hf) * 100))
        ax[1, k].add_patch(plt.Rectangle((-0.7, -0.7), 1.4, 1.4, fill=False, ec="red", lw=1.5, ls="--"))
        ax[1, k].set_xlabel("x [m]")
        ax[1, k].set_title("height map (red box: where the feet go)", fontsize=10)
        fig.colorbar(im, ax=ax[1, k], label="height [cm]", fraction=0.046)
    fig.suptitle("DRFloor-SchV3s floor mix (+ friction 0.2 polished ... 2.0 rubber mat, per robot)", fontsize=16)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
