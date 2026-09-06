"""训练潜空间世界模型（LeWM 方式，超参数见 conf/train_latent_wm.yaml）。

每步：encoder -> latent1（B, T, D）；历史 3 帧经时间 Transformer -> latent2；预测器 (latent2, 动作块) -> 下一帧 latent1；
损失 = pred MSE（不 detach）+ 0.09 * SIGReg（全部帧）。AdamW 5e-5 / 1e-3，线性 warmup + 余弦退火，梯度裁剪 1.0，bf16。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import gymnasium.spaces as spaces
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from pprl.data.wm_dataset import ColumnNormalizer, WindowDataset, collate_windows, load_episode, to_device
from pprl.envs import PointCloudSpace
from pprl.models.latent_world_model import LatentWorldModel


def build_model(config: DictConfig, point_dim: int, state_dim: int, action_dim: int) -> LatentWorldModel:
    obs_space = spaces.Dict({
        "points": PointCloudSpace(int(config.data.max_expected_num_points), -np.inf, np.inf, feature_shape=(point_dim,)),
        "state": spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32),
    })
    model_cfg = OmegaConf.create(OmegaConf.to_container(config.model, resolve=True))
    with open_dict(model_cfg):
        model_cfg.pop("name", None)
    encoder = instantiate(model_cfg, _convert_="partial", obs_space=obs_space)
    wm = config.wm
    return LatentWorldModel(
        encoder=encoder, action_dim=action_dim, history_size=int(wm.history_size), num_preds=int(wm.num_preds),
        temporal_depth=int(wm.temporal.depth), temporal_num_heads=int(wm.temporal.num_heads),
        temporal_mlp_ratio=float(wm.temporal.mlp_ratio), temporal_dropout=float(wm.temporal.dropout),
        predictor_hidden_sizes=list(wm.predictor.hidden_sizes), predictor_nonlinearity=str(wm.predictor.hidden_nonlinearity),
        sigreg_weight=float(wm.sigreg.weight), sigreg_knots=int(wm.sigreg.knots), sigreg_num_proj=int(wm.sigreg.num_proj),
    )


@torch.no_grad()
def evaluate(model, loader, device, bf16: bool) -> dict[str, float]:
    model.eval()
    sums, n = {}, 0
    for obs, action in loader:
        obs, action = to_device(obs, device), action.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            out = model(obs, action)
        b = action.shape[0]
        for k in ("loss", "pred_loss", "sigreg_loss"):
            sums[k] = sums.get(k, 0.0) + out[k].item() * b
        # 参考基线：直接用当前帧 latent1 当作下一帧的预测
        H, P = model.history_size, model.num_preds
        copy_last = (out["emb"][:, P - 1 : P - 1 + H] - out["emb"][:, P:]).pow(2).mean().item() if P == 1 else float("nan")
        sums["copy_last_pred_loss"] = sums.get("copy_last_pred_loss", 0.0) + copy_last * b
        n += b
    model.train()
    return {k: v / n for k, v in sums.items()}


@hydra.main(version_base=None, config_path="../conf", config_name="train_latent_wm")
def main(config: DictConfig) -> None:
    device = torch.device(config.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    seed = int(config.train.seed)
    torch.manual_seed(seed); np.random.seed(seed)
    out_dir = Path(config.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, out_dir / "config.yaml")

    # 数据：按回合划分训练/验证，动作标准化统计量取自训练集
    files = sorted(Path(config.data.dir).glob("ep_*.npz"))
    assert files, f"{config.data.dir} 里没有 ep_*.npz"
    if config.data.n_episodes:
        files = files[: int(config.data.n_episodes)]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n_train = int(len(files) * float(config.data.train_split))
    t0 = time.time()
    train_eps = [load_episode(files[i]) for i in perm[:n_train]]
    val_eps = [load_episode(files[i]) for i in perm[n_train:]]
    print(f"读取 {len(train_eps)} 训练回合 + {len(val_eps)} 验证回合（{time.time() - t0:.0f}s）")
    action_norm = ColumnNormalizer.fit(np.concatenate([ep["action"] for ep in train_eps], 0))
    num_steps = int(config.wm.history_size) + int(config.wm.num_preds)
    fs = int(config.data.frameskip)
    train_set = WindowDataset(train_eps, num_steps, fs, action_norm)
    val_set = WindowDataset(val_eps, num_steps, fs, action_norm)
    nw = int(config.train.num_workers)
    loader_kw = dict(batch_size=int(config.train.batch_size), num_workers=nw, collate_fn=collate_windows,
                     pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, generator=torch.Generator().manual_seed(seed), **loader_kw)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kw)
    print(f"子轨迹：训练 {len(train_set)}，验证 {len(val_set)}（每样本 {num_steps} 帧，frameskip {fs}）")

    # 模型
    ep0 = train_eps[0]
    model = build_model(config, ep0["points"].shape[1], ep0["state"].shape[1], ep0["action"].shape[1] * fs).to(device)
    if config.encoder_ckpt:
        sd = torch.load(config.encoder_ckpt, map_location="cpu")
        model.encoder.load_state_dict({k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")})
        print(f"encoder 权重来自 {config.encoder_ckpt}")
    n_params = {n: sum(p.numel() for p in m.parameters()) for n, m in [("encoder", model.encoder), ("temporal", model.temporal), ("predictor", model.predictor)]}
    print(f"latent 维度 {model.latent_dim}，动作块维度 {ep0['action'].shape[1] * fs}，参数量 {n_params}")

    # 优化器与调度：AdamW + 线性 warmup + 余弦退火（按 epoch）
    epochs, warmup = int(config.train.epochs), int(config.train.warmup_epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.train.lr), weight_decay=float(config.train.weight_decay))
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    bf16 = bool(config.train.bf16) and device.type == "cuda"

    metrics = open(out_dir / "metrics.jsonl", "w")
    step, t0 = 0, time.time()
    for epoch in range(epochs):
        for obs, action in train_loader:
            obs, action = to_device(obs, device), action.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                out = model(obs, action)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.train.grad_clip))
            optimizer.step()
            step += 1
            if step % int(config.train.log_interval) == 0:
                rec = {"step": step, "epoch": epoch, "train": {k: out[k].item() for k in ("loss", "pred_loss", "sigreg_loss")},
                       "lr": optimizer.param_groups[0]["lr"], "time": time.time() - t0}
                metrics.write(json.dumps(rec) + "\n"); metrics.flush()
                print(f"step {step} ep {epoch}: loss {rec['train']['loss']:.4f} pred {rec['train']['pred_loss']:.4f} sigreg {rec['train']['sigreg_loss']:.4f} ({rec['time']:.0f}s)")
        scheduler.step()
        val = evaluate(model, val_loader, device, bf16)
        metrics.write(json.dumps({"step": step, "epoch": epoch, "val": val, "time": time.time() - t0}) + "\n"); metrics.flush()
        print(f"[val] ep {epoch}: loss {val['loss']:.4f} pred {val['pred_loss']:.4f} (copy-last 基线 {val['copy_last_pred_loss']:.4f}) sigreg {val['sigreg_loss']:.4f}")
        ckpt = {"state_dict": model.state_dict(), "action_norm": action_norm.state_dict(),
                "dims": {"point_dim": int(ep0["points"].shape[1]), "state_dim": int(ep0["state"].shape[1]), "action_dim": int(ep0["action"].shape[1])},
                "config": OmegaConf.to_container(config, resolve=True), "epoch": epoch + 1}
        torch.save(ckpt, out_dir / f"weights_epoch_{epoch + 1}.pt")
        torch.save(ckpt, out_dir / "latent_wm.pt")
    metrics.close()
    print(f"完成，输出在 {out_dir}")


if __name__ == "__main__":
    main()
