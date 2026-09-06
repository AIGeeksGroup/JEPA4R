"""最小训练循环：合成数据 + AdamW，打印损失。用法：python train.py [model.K=256 train.steps=50 ...]"""

import sys
import time
from dataclasses import dataclass

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data import SyntheticSequences
from models import WorldModel, WorldModelConfig


@dataclass
class TrainConfig:
    env: str = "synthetic"  # 本次训练的环境名（一个 batch 只含一个环境）
    steps: int = 200
    batch_size: int = 4
    num_samples: int = 64
    lr: float = 3e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 10


def load_train_config(path: str = "configs/default.yaml", overrides=()) -> TrainConfig:
    cfg = OmegaConf.merge(OmegaConf.structured(TrainConfig), OmegaConf.load(path).train)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([o[len("train.") :] for o in overrides if o.startswith("train.")]))
    return OmegaConf.to_object(cfg)


def main(argv):
    path = "configs/default.yaml"
    model_cfg, train_cfg = WorldModelConfig.from_yaml(path, argv), load_train_config(path, argv)
    torch.manual_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("警告：没有可用的 CUDA，回退到 CPU（spconv 只有 Native 算法，很慢；只适合冒烟）")

    model = WorldModel(model_cfg).to(device)
    print(f"参数量 {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | K={model_cfg.K} D={model_cfg.D} T={model_cfg.T} C={model_cfg.C}")
    dataset = SyntheticSequences(
        train_cfg.num_samples, T=model_cfg.T, K=model_cfg.K, C=model_cfg.C,
        action_dim=model_cfg.action.envs[train_cfg.env].dim, seed=train_cfg.seed,
    )
    loader = DataLoader(dataset, batch_size=train_cfg.batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    model.train()
    step, t0 = 0, time.time()
    while step < train_cfg.steps:
        for points, actions in loader:
            out = model(points.to(device), actions.to(device), env=train_cfg.env)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            step += 1
            if step % train_cfg.log_every == 0 or step == 1:
                print(f"step {step:5d} | loss {out['loss'].item():.4f} | pred {out['pred_loss'].item():.4f} "
                      f"| reg {out['reg_loss'].item():.4f} | {time.time() - t0:.1f}s")
            if step >= train_cfg.steps:
                break


if __name__ == "__main__":
    main(sys.argv[1:])
