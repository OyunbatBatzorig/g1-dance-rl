"""Export an rsl-rl actor checkpoint (MLP 154-512-256-128-29, ELU, no obs normalizer) to ONNX for the Unitree
deploy stack, with the input/output names of Unitree's own export (obs [1,154] -> actions [1,29]).

  # 1. export (needs torch + onnx: env_isaaclab)
  conda run -n env_isaaclab python deploy/export_onnx.py export <model_N.pt> deploy/schuhplattler_v3s/exported/policy.onnx
  # 2. check ONNX == PyTorch (needs onnxruntime: s2s)
  conda run -n s2s python deploy/export_onnx.py check <model_N.pt> deploy/schuhplattler_v3s/exported/policy.onnx
"""
import sys

import numpy as np
import torch


def actor(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["actor_state_dict"]
    assert not any("normalizer" in k for k in sd), "checkpoint has an obs normalizer: export it too"
    dims = [sd["mlp.0.weight"].shape[1]] + [sd[f"mlp.{i}.weight"].shape[0] for i in (0, 2, 4, 6)]
    layers = []
    for j, i in enumerate((0, 2, 4, 6)):
        lin = torch.nn.Linear(dims[j], dims[j + 1])
        lin.weight.data, lin.bias.data = sd[f"mlp.{i}.weight"].float(), sd[f"mlp.{i}.bias"].float()
        layers += [lin] + ([torch.nn.ELU()] if j < 3 else [])
    return torch.nn.Sequential(*layers).eval(), dims


mode, ckpt, out = sys.argv[1:4]
net, dims = actor(ckpt)
if mode == "export":
    import onnx

    torch.onnx.export(net, torch.zeros(1, dims[0]), out, input_names=["obs"], output_names=["actions"],
                      opset_version=11, dynamo=False)
    onnx.checker.check_model(onnx.load(out))
    print(f"exported {out}: layers {dims}, obs [1,{dims[0]}] -> actions [1,{dims[-1]}]")
else:
    import onnxruntime as ort

    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    io = ([(i.name, i.shape) for i in sess.get_inputs()], [(o.name, o.shape) for o in sess.get_outputs()])
    x = np.random.default_rng(0).normal(0, 1, (4096, dims[0])).astype(np.float32)
    with torch.no_grad():
        y_t = net(torch.from_numpy(x)).numpy()
    y_o = np.concatenate([sess.run(None, {"obs": x[i:i + 1]})[0] for i in range(len(x))])
    print(f"{out}: io {io}")
    print(f"ONNX vs PyTorch on {len(x)} random obs: max |diff| = {np.abs(y_t - y_o).max():.2e} rad-scaled action")
