"""单模块形状测试 + 整体前向/反向。有 CUDA 就在 CUDA 上跑，否则回退到 CPU（spconv 的 Native 算法）。"""

import pytest
import torch

import losses.lewm as lewm_loss
from data import SyntheticSequences
from losses import LeWMLoss, LossConfig, SIGReg
from losses.lewm import match_motion_weights
from models import WorldModel, WorldModelConfig
from models.action import ActionConfig, ActionEncoder, EnvSpec
from models.aggregator import AggregatorConfig, SpatialAggregator
from models.ordering import Ordering, OrderingConfig, encode, hilbert_encode, z_order_encode
from models.predictor import Predictor, PredictorConfig
from models.temporal import TemporalConfig, TemporalEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 没有 CUDA 时全部在 CPU 上跑（spconv Native 算法）
B, T, K, C, D, A = 2, 3, 64, 6, 32, 7
ENV = "synthetic"


def small_cfg(**overrides) -> WorldModelConfig:
    cfg = WorldModelConfig(
        K=K, D=D, T=T, C=C,
        aggregator=AggregatorConfig(
            ordering=OrderingConfig(order="hilbert", grid_size=0.05, origin=[-1.0, -1.0, -1.0], depth=16),
            order=["z", "hilbert-trans"], stride=[2], enc_depths=[2, 1], enc_channels=[16, 32], enc_num_head=[2, 4],
            dec_depths=[1], dec_channels=[D], dec_num_head=[4], patch_size=16, drop_path=0.0,
        ),
        temporal=TemporalConfig(num_layers=1, num_heads=4),
        action=ActionConfig(envs={ENV: EnvSpec(dim=A, mean=[0.0] * A, std=[1.0] * A)}),
        predictor=PredictorConfig(num_layers=1, num_heads=4),
        loss=LossConfig(num_proj=64, chunk=16),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def unique_voxel_points(b, k, c, grid=0.05, seed=0):
    """每个点落在不同体素的随机点云 [b, k, c]（用来检验排列不变性）。"""
    g = torch.Generator().manual_seed(seed)
    cells = torch.stack([torch.randperm(40**3, generator=g)[:k] for _ in range(b)])  # 40 = 2 / grid
    xyz = torch.stack([cells // 1600, (cells // 40) % 40, cells % 40], dim=-1).float()
    xyz = -1.0 + (xyz + 0.5) * grid  # 体素中心
    return torch.cat([xyz, torch.rand(b, k, c - 3, generator=g)], dim=-1)


# ----------------------------------------------------------------------------- Ordering
def test_curve_codes_on_small_cube():
    depth = 2
    g = torch.arange(2**depth)
    cells = torch.cartesian_prod(g, g, g)  # 64 个格点
    assert z_order_encode(cells[:1] * 0 + torch.tensor([[1, 0, 0]]), depth=1).item() == 4
    assert z_order_encode(torch.tensor([[0, 1, 0]]), depth=1).item() == 2
    assert z_order_encode(torch.tensor([[0, 0, 1]]), depth=1).item() == 1
    for fn in (z_order_encode, hilbert_encode):
        code = fn(cells, depth)
        assert code.shape == (64,) and code.unique().numel() == 64 and code.min() == 0 and code.max() == 63
    walk = cells[torch.argsort(hilbert_encode(cells, depth))]
    assert torch.all((walk[1:] - walk[:-1]).abs().sum(-1) == 1)  # Hilbert 曲线相邻格点相邻
    for order in ("z", "z-trans", "hilbert", "hilbert-trans"):
        assert encode(cells, torch.ones(64, dtype=torch.long), depth, order).min() >= 1 << (depth * 3)


def test_ordering_shape_stable_and_permutation_invariant():
    ordering = Ordering(OrderingConfig(order="hilbert", grid_size=0.05))
    pts = unique_voxel_points(B, K, C)[..., :3]
    order = ordering(pts)
    assert order.shape == (B, K) and order.dtype == torch.long
    assert torch.equal(order.sort(dim=1).values, torch.arange(K).expand(B, K))  # 是排列
    perm = torch.randperm(K)
    a = pts.gather(1, order.unsqueeze(-1).expand(-1, -1, 3))
    b = pts[:, perm].gather(1, ordering(pts[:, perm]).unsqueeze(-1).expand(-1, -1, 3))
    assert torch.equal(a, b)  # 输入次序无意义：重排后的坐标序列相同
    dup = pts.clone()
    dup[:, 1] = dup[:, 0]  # 同体素的两点：稳定排序保持输入次序
    o = ordering(dup)
    for i in range(B):
        pos = {int(idx): j for j, idx in enumerate(o[i])}
        assert pos[0] + 1 == pos[1]
    grid = Ordering(OrderingConfig(grid_size=0.5, depth=16)).voxelize(torch.tensor([[[-5.0, 0.0, 1e6]]]))
    assert grid.dtype == torch.int32 and grid.tolist() == [[[0, 2, 2**16 - 1]]]  # 截断到 [0, 2**depth - 1]


# ----------------------------------------------------------------------------- SpatialAggregator
def test_aggregator_shapes_order_and_invariances():
    cfg = small_cfg()
    agg = SpatialAggregator(cfg.aggregator, in_channels=C, out_channels=D).to(DEVICE).eval()
    pts = unique_voxel_points(B, K, C).to(DEVICE)
    with torch.no_grad():
        tokens, positions = agg(pts)
        assert tokens.shape == (B, K, D) and positions.shape == (B, K, 3)
        code = encode(agg.ordering.voxelize(positions).reshape(-1, 3), depth=16, order="hilbert").view(B, K)
        assert torch.all(code[:, 1:] >= code[:, :-1])  # 输出已按曲线序排列
        tokens2, positions2 = agg(pts)
        assert torch.equal(tokens, tokens2) and torch.equal(positions, positions2)  # 评估时确定
        perm = torch.randperm(K, device=DEVICE)
        tokens3, positions3 = agg(pts[:, perm])
        assert torch.equal(positions, positions3) and torch.allclose(tokens, tokens3, atol=1e-4)  # 点的存储顺序无意义
        # 体素去重：同体素的点得到同一个向量
        dup = pts.clone()
        dup[:, 1, :3] = dup[:, 0, :3] + 0.001
        tok, pos = agg(dup)
        for i in range(B):
            j0, j1 = (pos[i] == dup[i, 0, :3]).all(-1).nonzero()[0], (pos[i] == dup[i, 1, :3]).all(-1).nonzero()[0]
            assert torch.allclose(tok[i, j0], tok[i, j1], atol=1e-5)


def test_aggregator_rejects_wrong_channels():
    cfg = small_cfg()
    agg = SpatialAggregator(cfg.aggregator, in_channels=C, out_channels=D)
    with pytest.raises(AssertionError):
        agg(torch.rand(B, K, C + 1))
    with pytest.raises(AssertionError):
        SpatialAggregator(cfg.aggregator, in_channels=C, out_channels=D + 1)  # dec_channels[0] != D


# ----------------------------------------------------------------------------- TemporalEncoder
@pytest.mark.parametrize("readout", ["last_frame", "query"])
@pytest.mark.parametrize("enc", ["learned", "sinusoidal"])
def test_temporal_encoder(readout, enc):
    cfg = TemporalConfig(num_layers=1, num_heads=4, readout=readout, pos_encoding=enc, time_encoding=enc, causal=True)
    temporal = TemporalEncoder(cfg, K=K, T=T, D=D)
    latent2 = temporal(torch.randn(B, T, K, D))
    assert latent2.shape == (B, K, D)
    mask = temporal.causal_mask
    frame = torch.arange(T).repeat_interleave(K)
    assert mask.shape == (T * K, T * K) and torch.equal(mask, frame[None, :] > frame[:, None])
    with pytest.raises(AssertionError):
        temporal(torch.randn(B, T + 1, K, D))


# ----------------------------------------------------------------------------- ActionEncoder / Predictor
def test_action_encoder():
    enc = ActionEncoder(ActionConfig(envs={ENV: EnvSpec(dim=A, mean=[0.0] * A, std=[1.0] * A), "other": EnvSpec(dim=3, mean=[0.0] * 3, std=[1.0] * 3)}), D=D)
    assert enc(torch.randn(B, 1, A), ENV).shape == (B, 1, D)
    assert enc(torch.randn(B, 5, 3), "other").shape == (B, 5, D)
    with pytest.raises(AssertionError):
        enc(torch.randn(B, 1, A), "other")
    with pytest.raises(AssertionError):
        enc(torch.randn(B, 1, A), "unknown")


def test_action_encoder_output_scale_is_explicit():
    base_cfg = ActionConfig(envs={ENV: EnvSpec(dim=A, mean=[0.0] * A, std=[1.0] * A)})
    scaled_cfg = ActionConfig(envs=base_cfg.envs, output_scale=4.0)
    base = ActionEncoder(base_cfg, D=D)
    scaled = ActionEncoder(scaled_cfg, D=D)
    scaled.load_state_dict(base.state_dict())
    actions = torch.randn(B, 2, A)
    assert torch.allclose(scaled.encode_chunks(actions, ENV), 4.0 * base.encode_chunks(actions, ENV))


@pytest.mark.parametrize("fusion", ["add", "modulate"])
def test_predictor(fusion):
    pred_net = Predictor(PredictorConfig(num_layers=2, num_heads=4, fusion=fusion), D=D, envs=[ENV])
    latent2, action = torch.randn(B, K, D), torch.randn(B, 1, D, requires_grad=True)
    out = pred_net(latent2, action, ENV)
    assert out.shape == (B, K, D)
    out.sum().backward()
    assert action.grad is not None and action.grad.abs().sum() > 0  # 动作条件确实进入了预测
    with pytest.raises(AssertionError):
        pred_net(latent2, torch.randn(B, 2, D), ENV)


# ----------------------------------------------------------------------------- 损失
def test_sigreg_chunked_matches_lewm():
    P, num_proj = 37, 64
    sig = SIGReg(knots=17, num_proj=num_proj, chunk=5)
    proj = torch.randn(P, B, D)
    torch.manual_seed(0)
    A_ = torch.randn(D, num_proj)
    A_ = A_.div_(A_.norm(p=2, dim=0))
    x_t = (proj @ A_).unsqueeze(-1) * sig.t  # LeWM module.py 原式
    err = (x_t.cos().mean(-3) - sig.phi).square() + x_t.sin().mean(-3).square()
    reference = ((err @ sig.weights) * proj.size(-2)).mean()
    torch.manual_seed(0)
    assert torch.allclose(sig(proj), reference, atol=1e-6)


def test_lewm_loss():
    crit = LeWMLoss(LossConfig(num_proj=64, chunk=16))
    latent1_all = torch.randn(B, T + 1, K, D, requires_grad=True)
    pred = torch.randn(B, K, D, requires_grad=True)
    out = crit(pred, latent1_all[:, -1], latent1_all)
    assert set(out) == {"loss", "pred_loss", "latent_loss", "flow_loss", "rank_loss", "reg_loss"} and out["loss"].shape == ()
    out["loss"].backward()
    assert latent1_all.grad is not None and latent1_all.grad[:, -1].abs().sum() > 0  # 目标不截断梯度


def test_lewm_loss_supports_all_transition_targets():
    crit = LeWMLoss(LossConfig(num_proj=16, chunk=8))
    latent1_all = torch.randn(B, T + 1, K, D, requires_grad=True)
    pred = torch.randn(B, T, K, D, requires_grad=True)
    out = crit(pred, latent1_all[:, 1:], latent1_all)
    assert out["pred_loss"].shape == () and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert pred.grad is not None and latent1_all.grad is not None


def test_motion_matching_upweights_displaced_patch_and_combines_latent_loss():
    source = torch.tensor([[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]]).expand(2, -1, -1, -1)
    target_pos = torch.tensor([[[[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]]]).expand(2, -1, -1, -1)
    match, weights = match_motion_weights(
        source, target_pos, threshold=0.1, temperature=0.01, floor=0.25, gamma=1.0
    )
    assert torch.equal(match, torch.tensor([[[0, 1]], [[0, 1]]]))
    assert torch.allclose(weights.mean(dim=-1), torch.ones(2, 1))
    assert torch.all(weights[..., 1] > weights[..., 0])

    latent1_all = torch.zeros(2, 2, 2, 1, requires_grad=True)
    target = latent1_all[:, 1:].clone()
    target[:, :, 1] = 2.0
    pred = torch.zeros_like(target, requires_grad=True)
    crit = LeWMLoss(LossConfig(reg_weight=0.0, flow_weight=1.0, num_proj=8, chunk=4,
                               motion_threshold=0.1, motion_temperature=0.01))
    out = crit(pred, target, latent1_all, torch.cat([source, target_pos], dim=1))
    assert out["flow_loss"] > out["latent_loss"]
    assert out["latent_loss"] < out["pred_loss"] < out["flow_loss"]


def test_action_ranking_penalizes_indistinguishable_negative_prediction():
    positions = torch.tensor(
        [[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]]
    ).expand(2, -1, -1, -1)
    latent1_all = torch.ones(2, 2, 2, 1, requires_grad=True)
    target = latent1_all[:, 1:]
    correct = torch.zeros_like(target, requires_grad=True)
    indistinguishable = torch.zeros_like(target, requires_grad=True)
    worse = torch.full_like(target, 3.0, requires_grad=True)
    criterion = LeWMLoss(
        LossConfig(reg_weight=0.0, flow_weight=0.0, ranking_weight=1.0, ranking_margin=0.1,
                   num_proj=8, chunk=4)
    )
    ranked = criterion(correct, target, latent1_all, positions, negative_pred=indistinguishable)
    separated = criterion(correct, target, latent1_all, positions, negative_pred=worse)
    assert ranked["rank_loss"].item() == pytest.approx(0.1)
    assert separated["rank_loss"].item() == pytest.approx(0.0)
    ranked["loss"].backward()
    assert correct.grad is not None and indistinguishable.grad is not None


def test_unweighted_action_ranking_does_not_use_motion_matching(monkeypatch):
    latent1_all = torch.ones(2, 2, 2, 1, requires_grad=True)
    target = latent1_all[:, 1:]
    correct = torch.zeros_like(target, requires_grad=True)
    negative = torch.zeros_like(target, requires_grad=True)
    criterion = LeWMLoss(
        LossConfig(
            reg_weight=0.0,
            flow_weight=0.0,
            ranking_weight=1.0,
            ranking_margin=0.1,
            ranking_motion_weighted=False,
            num_proj=8,
            chunk=4,
        )
    )
    monkeypatch.setattr(
        lewm_loss,
        "match_motion_weights",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("motion matching was called")),
    )

    output = criterion(correct, target, latent1_all, positions_all=None, negative_pred=negative)

    assert output["rank_loss"].item() == pytest.approx(0.1)
    output["loss"].backward()
    assert correct.grad is not None and negative.grad is not None


def test_motion_weighted_action_ranking_still_requires_positions():
    latent1_all = torch.ones(2, 2, 2, 1)
    target = latent1_all[:, 1:]
    prediction = torch.zeros_like(target)
    criterion = LeWMLoss(
        LossConfig(reg_weight=0.0, ranking_weight=1.0, ranking_motion_weighted=True, num_proj=8, chunk=4)
    )

    with pytest.raises(ValueError, match="positions_all"):
        criterion(prediction, target, latent1_all, positions_all=None, negative_pred=prediction)


# ----------------------------------------------------------------------------- WorldModel
def test_world_model_forward_encode_rollout():
    cfg = small_cfg()
    model = WorldModel(cfg).to(DEVICE).eval()
    ds = SyntheticSequences(4, T=T, K=K, C=C, action_dim=A)
    points, actions = ds[0]
    assert points.shape == (T + 1, K, C) and actions.shape == (T, A)
    points = torch.stack([ds[i][0] for i in range(B)]).to(DEVICE)
    actions = torch.stack([ds[i][1] for i in range(B)]).to(DEVICE)
    with torch.no_grad():
        out = model(points, actions, env=ENV)
        assert {"pred", "target", "loss", "latent1", "latent2"} <= set(out)
        assert out["pred"].shape == out["target"].shape == (B, K, D)
        assert out["pred_all"].shape == (B, T, K, D)
        assert out["latent1"].shape == (B, T, K, D) and out["latent2"].shape == (B, K, D)
        assert out["loss"].shape == () and torch.isfinite(out["loss"])
        latent1, latent2 = model.encode(points[:, :T])
        assert latent1.shape == (B, T, K, D) and latent2.shape == (B, K, D)
        assert torch.allclose(latent2, out["latent2"], atol=1e-5)
        H = 5
        rollout_actions = torch.randn(B, H, A, device=DEVICE)
        context_actions = torch.randn(B, T - 1, A, device=DEVICE)
        assert model.rollout(
            points[:, :T], rollout_actions, context_actions=context_actions
        ).shape == (B, H, K, D)
        assert model.rollout(points[:, :T], rollout_actions).shape == (B, H, K, D)
    with pytest.raises(AssertionError):
        model(points[:, :T], actions, env=ENV)
    with pytest.raises(AssertionError):
        model(points, actions, env="unknown")


def test_world_model_skip_frame_consumes_atomic_action_chunks():
    cfg = small_cfg(skip_frame=2)
    model = WorldModel(cfg).to(DEVICE).eval()
    points = unique_voxel_points(B * (T + 1), K, C).view(B, T + 1, K, C).to(DEVICE)
    actions = torch.randn(B, T * 2, A, device=DEVICE)
    with torch.no_grad():
        out = model(points, actions, env=ENV)
        rollout_actions = torch.randn(B, 4 * 2, A, device=DEVICE)
        context_actions = torch.randn(B, (T - 1) * 2, A, device=DEVICE)
        rollout = model.rollout(
            points[:, :T], rollout_actions, env=ENV, context_actions=context_actions
        )
    assert out["pred_all"].shape == (B, T, K, D)
    assert rollout.shape == (B, 4, K, D)


def test_world_model_cross_attention_accepts_negative_actions_for_ranking():
    cfg = small_cfg(skip_frame=2)
    cfg.predictor.fusion = "cross_attention"
    cfg.loss.ranking_weight = 1.0
    cfg.loss.ranking_margin = 0.1
    model = WorldModel(cfg).to(DEVICE).train()
    points = unique_voxel_points(B * (T + 1), K, C).view(B, T + 1, K, C).to(DEVICE)
    actions = torch.randn(B, T * 2, A, device=DEVICE)
    output = model(points, actions, env=ENV, negative_actions=actions.roll(1, dims=0))
    assert output["rank_loss"].shape == () and torch.isfinite(output["rank_loss"])
    output["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.action_encoder.parameters())


def test_world_model_backward():
    cfg = small_cfg()
    model = WorldModel(cfg).to(DEVICE).train()
    ds = SyntheticSequences(4, T=T, K=K, C=C, action_dim=A)
    points = torch.stack([ds[i][0] for i in range(B)]).to(DEVICE)
    actions = torch.stack([ds[i][1] for i in range(B)]).to(DEVICE)
    out = model(points, actions, env=ENV)
    assert out["target"].requires_grad  # 目标不截断梯度
    out["loss"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for sub in (model.aggregator, model.temporal, model.action_encoder, model.predictor):
        assert any(p.grad.abs().sum() > 0 for p in sub.parameters()), type(sub).__name__


def test_default_config_builds():
    cfg = WorldModelConfig.from_yaml("configs/default.yaml", ["model.K=256", "train.steps=1"])
    assert isinstance(cfg, WorldModelConfig) and cfg.K == 256 and cfg.aggregator.dec_channels[0] == cfg.D
    model = WorldModel(cfg)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_world_model_action_at_each_time_step_changes_matching_prediction():
    cfg = small_cfg()
    model = WorldModel(cfg).to(DEVICE).eval()
    ds = SyntheticSequences(4, T=T, K=K, C=C, action_dim=A)
    points = torch.stack([ds[i][0] for i in range(B)]).to(DEVICE)
    actions = torch.stack([ds[i][1] for i in range(B)]).to(DEVICE)
    changed = actions.clone()
    changed[:, 0] += 5.0
    with torch.no_grad():
        baseline = model(points, actions, env=ENV)
        altered = model(points, changed, env=ENV)
    assert not torch.allclose(baseline["pred_all"][:, 0], altered["pred_all"][:, 0])
