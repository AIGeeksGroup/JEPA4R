"""LeJEPA Tiny/14 training on Cube with LeJEPA's own recipe and training loop."""

import argparse
import json
import os
import time
from pathlib import Path
import stable_worldmodel as swm
import torch
import torch.nn as nn
from stable_pretraining.backbone.utils import vit_hf
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import MLP
from torchvision.transforms import v2


class SIGReg(nn.Module):  # LeJEPA's MINIMAL.md, as is
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device="cuda")
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class Encoder(nn.Module):
    """HF ViT-tiny/14 (LeWM's) + a 512-d head; proj is LeJEPA's projector (used in the training loss only)."""

    def __init__(self, proj_dim=16):
        super().__init__()
        self.backbone = vit_hf(size="tiny", patch_size=14, image_size=224, pretrained=False, use_mask_token=False)
        self.backbone.gradient_checkpointing_enable()  # memory only, same computation (1024 images of 224 per batch)
        self.head = nn.Linear(192, 512)
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def embed(self, x):
        return self.head(self.backbone(x, interpolate_pos_encoding=True).last_hidden_state[:, 0])

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.embed(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


class Frames(Dataset):
    """Every frame of the Cube dataset (uint8 [3, 224, 224]), V views by the official augmentations."""

    def __init__(self, name, V):
        self.ds = swm.data.load_dataset(name, transform=None, num_steps=1, frameskip=1, keys_to_load=["pixels"])
        self.V = V
        self.aug = v2.Compose(
            [
                v2.RandomResizedCrop(224, scale=(0.08, 1.0)),
                v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                v2.RandomGrayscale(p=0.2),
                v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
                v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img = self.ds[i]["pixels"][0]  # uint8 [3, 224, 224]
        return torch.stack([self.aug(img) for _ in range(self.V)])


def main(*, gate=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="current Cube training label")
    ap.add_argument("--dataset", required=True, help="explicit paired Cube RGB training view")
    ap.add_argument("--name", default=None, help="checkpoint directory name (default: lejepa_<task>)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--V", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lamb", type=float, default=0.02)
    ap.add_argument("--proj-dim", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--max-steps", type=int, default=0, help="smoke test: stop after this many steps, no checkpoint (0 = no limit)"
    )
    args = ap.parse_args()
    torch.manual_seed(0)

    out = Path(os.environ["STABLEWM_HOME"]) / "checkpoints" / (args.name or f"lejepa_{args.task}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({"algo": "lejepa", "encoder": "vit_hf tiny p14 224 + Linear(192,512)", **vars(args)}, indent=1)
    )
    if gate is None:
        ds = Frames(args.dataset, args.V)
        train = DataLoader(
            ds,
            batch_size=args.bs,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
            persistent_workers=True,
            prefetch_factor=4,
        )
    else:
        train = gate.loader(args)
        ds = train.dataset
    net = Encoder(proj_dim=args.proj_dim).to("cuda")
    sigreg = SIGReg().to("cuda")
    opt = torch.optim.AdamW([{"params": net.parameters(), "lr": args.lr, "weight_decay": 5e-2}])
    warmup_steps, total_steps = len(train), len(train) * args.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
    scaler = GradScaler(enabled=True)
    start_epoch = 0 if gate is None else gate.attach(net, sigreg, opt, scheduler, scaler)
    print(f"{args.task}: {len(ds)} frames, {len(train)} steps per epoch, {args.epochs} epochs", flush=True)
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        net.train()
        start_batch = 0 if gate is None else gate.next_batch_index(epoch)
        for step, vs in enumerate(train, start=start_batch):
            if gate is not None:
                gate.before_step(net)
            with autocast("cuda", dtype=torch.bfloat16) if gate is None else gate.autocast():
                vs = vs.to("cuda", non_blocking=True)
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                loss = sigreg_loss * args.lamb + inv_loss * (1 - args.lamb)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            if gate is not None and gate.after_step(
                epoch, step, emb=emb, proj=proj, inv_loss=inv_loss, sigreg_loss=sigreg_loss, loss=loss
            ):
                return
            if args.max_steps and step + 1 >= args.max_steps:
                print(
                    f"smoke test: {step + 1} steps, {time.time() - t0:.1f}s, loss {loss.item():.4f}, peak GPU memory "
                    f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
                    flush=True,
                )
                return
            if step % 500 == 0:
                print(
                    f"epoch {epoch} step {step}/{len(train)} loss {loss.item():.4f} sigreg {sigreg_loss.item():.4f} "
                    f"inv {inv_loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s",
                    flush=True,
                )
        torch.save(
            {"backbone": net.backbone.state_dict(), "head": net.head.state_dict(), "proj": net.proj.state_dict()},
            out / f"weights_epoch_{epoch + 1}.pt",
        )
        print(f"epoch {epoch + 1} saved, {time.time() - t0:.0f}s in total", flush=True)
