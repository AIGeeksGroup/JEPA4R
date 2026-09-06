"""Run the untouched upstream CLI with a CPU-only hardware-logging shim.

Upstream train_rl calls torch.cuda.current_device even for --num-gpus 0.
The shared local configuration selects PointNetLastIdentity. Rollout, PPO and
checkpoint implementations remain upstream; the CPU shim affects logging.
"""
import os
import json
import importlib
from pathlib import Path
import runpy
import sys
import time

import numpy as np
import torch

if not torch.cuda.is_available():
    # A CPU wheel has no current CUDA device. Upstream's unconditional logging
    # argument needs a sentinel; the replacement logger below ignores it.
    torch.cuda.current_device = lambda: 0
    import maniskill2_learn.utils.torch as upstream_torch

    def cpu_hardware_log(device=None, unit="G", number_only=True):
        return {"model_device": "cpu"}

    upstream_torch.get_cuda_info = cpu_hardware_log

root = Path(__file__).resolve().parents[1]

if "--evaluation" not in sys.argv and "--eval" not in sys.argv:
    train_module = importlib.import_module("maniskill2_learn.apis.train_rl")
    original_train = train_module.train_rl

    def traced_train(*args, **kwargs):
        agent, _, _, replay = args[:4]
        started = time.time()
        visual = agent.actor.backbone.visual_nn
        parts = {"encoder": visual, "actor_head": agent.actor.backbone.final_mlp,
                 "critic_head": agent.critic.values[0].backbone.final_mlp}
        before = {key: [p.detach().clone() for p in module.parameters()] for key, module in parts.items()}
        observed_shapes = {}

        def inspect_encoder(module, inputs, output):
            if not observed_shapes:
                observed_shapes.update(input={key: list(value.shape) for key, value in inputs[0].items()},
                                       output=list(output.shape))

        hook = visual.register_forward_hook(inspect_encoder)
        try:
            result = original_train(*args, **kwargs)
        finally:
            hook.remove()
        changed = {key: sum(float((p.detach() - old).abs().sum())
                            for p, old in zip(module.parameters(), before[key]))
                   for key, module in parts.items()}
        pointnet = (visual.encoder.frame_encoder.pointnet
                    if hasattr(visual, "encoder") and hasattr(visual.encoder, "frame_encoder") else visual)
        report = dict(pid=os.getpid(), elapsed_seconds=time.time()-started, model_device=str(agent.device),
                      transition_count=len(replay), encoder_shapes=observed_shapes,
                      encoder_class=f"{type(visual).__module__}.{type(visual).__name__}",
                      encoder_activations=[type(block.act).__name__ for block in pointnet.conv.mlp],
                      encoder_frozen=not any(p.requires_grad for p in visual.parameters()),
                      parameter_l1_change=changed, finite_parameters=all(bool(torch.isfinite(p).all()) for p in agent.parameters()),
                      validation_kind="bounded smoke; does not establish convergence")
        output = Path(kwargs["work_dir"])
        (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        memory = replay.memory
        np.savez_compressed(output / "real_transitions.npz", xyz=memory["obs"]["xyz"],
                            rgb=memory["obs"]["rgb"], actions=memory["actions"],
                            next_xyz=memory["next_obs"]["xyz"], next_rgb=memory["next_obs"]["rgb"])
        print(json.dumps(report, indent=2), flush=True)
        return result

    train_module.train_rl = traced_train

entrypoint = root / "vendor/ManiSkill2-Learn/maniskill2_learn/apis/run_rl.py"
sys.argv[0] = str(entrypoint)
print(f"Isolated official RL launcher PID={os.getpid()} device=cpu", flush=True)
runpy.run_path(str(entrypoint), run_name="__main__")
