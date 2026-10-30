def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, help='mw_binpick / mw_assembly / mw_button / mw_drawer / mw_hammer')
    ap.add_argument('--dataset', default=None, help='dataset under datasets/ (default: metaworld/<task>.h5)')
    ap.add_argument('--name', default=None, help='checkpoint directory name (default: lejepa_<task>)')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--V', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--lamb', type=float, default=0.02)
    ap.add_argument('--proj-dim', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-steps', type=int, default=0, help='smoke test: stop after this many steps, no checkpoint (0 = no limit)')
    ap.add_argument('--resume-model', type=str, default=None)
    ap.add_argument('--start-epoch', type=int, default=0)
    ap.add_argument('--verified-export-root', type=str, default=None)
    args = ap.parse_args()
    _validate_resume_args(args)
    torch.manual_seed(0)

    out = Path(os.environ['STABLEWM_HOME']) / 'checkpoints' / (args.name or f'lejepa_{args.task}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps({'algo': 'lejepa', 'encoder': 'vit_hf tiny p14 224 + Linear(192,512)',
                                                 **vars(args)}, indent=1))
    ds = MWFrames(args.dataset or f'metaworld/{args.task}.h5', args.V)
    train = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True, num_workers=args.workers,
                       persistent_workers=True, prefetch_factor=4)
    net = Encoder(proj_dim=args.proj_dim).to('cuda')
    sigreg = SIGReg().to('cuda')
    _parent_receipt = _load_resume_model(net, args, out, _RESUME_BINDING)
    opt = torch.optim.AdamW([{'params': net.parameters(), 'lr': args.lr, 'weight_decay': 5e-2}])
    warmup_steps, total_steps = len(train), len(train) * args.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])
    scaler = GradScaler(enabled=True)
    _first_update = _prepare_resume(scheduler, opt, net, args, len(train), out, _parent_receipt, _RESUME_BINDING)
    print(f'{args.task}: {len(ds)} frames, {len(train)} steps per epoch, {args.epochs} epochs', flush=True)
    t0 = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        net.train()
        for step, vs in enumerate(train):
            with autocast('cuda', dtype=torch.bfloat16):
                vs = vs.to('cuda', non_blocking=True)
                emb, proj = net(vs)
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                loss = sigreg_loss * args.lamb + inv_loss * (1 - args.lamb)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            _record_first_update(_first_update, scheduler, out)
            if args.max_steps and step + 1 >= args.max_steps:
                print(f'smoke test: {step + 1} steps, {time.time() - t0:.1f}s, loss {loss.item():.4f}, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB', flush=True)
                _smoke_export(net, out, args, step + 1)
                return
            if step % 500 == 0:
                print(f'epoch {epoch} step {step}/{len(train)} loss {loss.item():.4f} sigreg {sigreg_loss.item():.4f} '
                      f'inv {inv_loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s', flush=True)
        _save_epoch(net, out, epoch + 1, args.verified_export_root)
        print(f'epoch {epoch + 1} saved, {time.time() - t0:.0f}s in total', flush=True)
