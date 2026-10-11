"""CPU contracts for explicit visual/state fusion; no benchmark or model download."""
import hashlib
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from ogbench_il_rl.data import dataset as data
from ogbench_il_rl.data.dataset_pcd import CubeOfficialPCDDataset, _PCDSliceView
from ogbench_il_rl.data.state_fusion import CubeStateVisualDataset
from ogbench_il_rl.evaluation.native_single_frame import SingleFrameAgent
from ogbench_il_rl.models import state_fusion as fusion
from ogbench_il_rl.models.policy import make_optimizer, unit_gaussian_nll
from ogbench_il_rl.tests.test_policy import state_hash, world


class DinoFixture(nn.Module):
    def __init__(self, model_folder):
        super().__init__()
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(12, 384), nn.BatchNorm1d(384))
        self.calls = 0
        self.requires_grad_(False).train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, inputs):
        self.calls += 1
        with torch.no_grad():
            return SimpleNamespace(logits=self.trunk(inputs))


@pytest.mark.parametrize('route,latent', [('pixels', 192), ('pcd', 192), ('dino', 384)])
def test_fusion_concat_mask_frozen_update_and_strict_reload(route, latent, monkeypatch):
    torch.set_num_threads(4)
    torch.manual_seed(21)
    if route == 'dino':
        from ogbench_il_rl.models import frozen_dino
        monkeypatch.setattr(frozen_dino, 'FrozenDinoCLS', DinoFixture)
        factory = dict(model_folder='fixture-only', modality='pixels', latent_dim=384)
    else:
        factory = dict(world_model=world(), modality=route)
    policy = fusion.create_policy(**factory, goal_state_mode='object_xyz', seed=0)
    assert isinstance(policy, fusion.FrozenStateVisualGCBC)
    assert policy.action_head[0].in_features == 2 * (latent + 28)
    current = torch.randn(4, 2, 6) if route == 'pcd' else torch.randn(4, 3, 2, 2)
    goal = torch.randn_like(current)
    current_state = torch.arange(112, dtype=torch.float32).reshape(4, 28)
    goal_state = current_state + 200
    originals = [value.clone() for value in (current, goal, current_state, goal_state)]
    masked = torch.zeros_like(goal_state)
    masked[:, 19:22] = goal_state[:, 19:22]
    expected = torch.cat((policy.encoder(current).logits, current_state,
                          policy.encoder(goal).logits, masked), dim=-1)
    seen = []
    hook = policy.action_head[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    mean = policy(current, goal, current_state, goal_state)
    hook.remove()
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
    assert mean.shape == (4, 5) and torch.isfinite(mean).all()
    for actual, original in zip((current, goal, current_state, goal_state), originals):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    outside = goal_state.clone()
    outside[:, :19] += 1000
    outside[:, 22:] -= 1000
    torch.testing.assert_close(policy(current, goal, current_state, outside), mean, rtol=0, atol=0)

    frozen, head = state_hash(policy.encoder), state_hash(policy.action_head)
    counter = policy.encoder if route == 'dino' else policy.encoder.encoder
    calls = counter.calls
    policy.train()
    assert not policy.encoder.training
    assert all(not module.training for module in policy.encoder.modules() if isinstance(module, nn.BatchNorm1d))
    optimizer = make_optimizer(policy)
    unit_gaussian_nll(policy(current, goal, current_state, goal_state), torch.zeros(4, 5)).backward()
    assert counter.calls == calls + 2
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in policy.encoder.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in policy.action_head.parameters())
    optimizer.step()
    assert state_hash(policy.encoder) == frozen
    assert state_hash(policy.action_head) != head
    restored = fusion.create_policy(**factory, goal_state_mode='object_xyz', seed=99)
    restored.load_state_dict(policy.state_dict(), strict=True)
    torch.testing.assert_close(restored(current, goal, current_state, goal_state),
                               policy(current, goal, current_state, goal_state), rtol=0, atol=0)


def test_default_full_state_and_invalid_states():
    policy = fusion.create_policy(world_model=world())
    assert policy.goal_state_mode == 'full' and policy.latent_dim == 192
    current, goal = torch.randn(2, 3, 2, 2), torch.randn(2, 3, 2, 2)
    state = torch.arange(56, dtype=torch.float32).reshape(2, 28)
    seen = []
    hook = policy.action_head[0].register_forward_pre_hook(lambda module, args: seen.append(args[0].detach().clone()))
    policy.get_action(dict(pixels=current, goal=goal, current_state=state, goal_state=state + 100))
    hook.remove()
    torch.testing.assert_close(seen[0][:, 192:220], state, rtol=0, atol=0)
    torch.testing.assert_close(seen[0][:, 412:], state + 100, rtol=0, atol=0)
    with pytest.raises(ValueError):
        policy.get_action(dict(pixels=current, goal=goal))
    for invalid in (state[:, :3], state.double(), torch.full_like(state, float('nan'))):
        with pytest.raises(ValueError):
            policy(current, goal, state, invalid)


class SliceReader:
    def __init__(self, episode, pcd=False):
        self.episode, self.pcd = episode, pcd
        self.row_offset, self.terminal = 0, False
        self.h5_file = None

    def _load_slice(self, episode, start, stop):
        steps = torch.arange(start, stop)
        result = dict(action=torch.zeros(stop - start, 5), source_row=self.episode * 1001 + steps + self.row_offset,
                      source_episode=torch.full((stop - start,), self.episode), source_step=steps,
                      source_terminal=torch.full((stop - start,), self.terminal))
        result['pcd' if self.pcd else 'pixels'] = torch.zeros(stop - start, 1200, 6) if self.pcd else torch.zeros(stop - start, 3, 2, 2)
        return result


@pytest.mark.parametrize('modality', ['pixels', 'pcd'])
def test_state_rows_boundaries_wrong_source_terminal_and_pickle(tmp_path, monkeypatch, modality):
    index_path = tmp_path / 'index.json'
    index_path.write_text(json.dumps(dict(status='passed', exit_code=0,
        train=[dict(episode=0, path='fixture0'), dict(episode=1, path='fixture1')])))
    cls = data.CubeOfficialDataset if modality == 'pixels' else CubeOfficialPCDDataset
    readers = [SliceReader(episode, modality == 'pcd') for episode in range(2)]
    monkeypatch.setattr(cls, '_reader', lambda self, episode: _PCDSliceView(readers[episode]) if modality == 'pcd' else readers[episode])
    visual = cls(index_path, 'train', None)
    folder = tmp_path / 'train'
    folder.mkdir()
    path = folder / 'observation.npy'
    states = np.arange(2002 * 28, dtype=np.float32).reshape(2002, 28)
    np.save(path, states)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = CubeStateVisualDataset(visual, path, expected_sha256=sha)
    for pair, rows in [((0, 1), (0, 1)), ((998, 999), (998, 999)),
                       ((999, 999), (999, 999)), ((1000, 1001), (1001, 1002))]:
        result = dataset[pair]
        assert (int(result['source_row']), int(result['goal_source_row'])) == rows
        np.testing.assert_array_equal(result['current_state'], states[rows[0]])
        np.testing.assert_array_equal(result['goal_state'], states[rows[1]])
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._observations is None
    np.testing.assert_array_equal(restored[(1000, 1001)]['goal_state'], states[1002])
    with pytest.raises(ValueError):
        CubeStateVisualDataset(visual, path, expected_sha256='0' * 64)
    readers[0].row_offset = 1001
    with pytest.raises(ValueError):
        dataset[(0, 1)]
    readers[0].row_offset, readers[0].terminal = 0, True
    with pytest.raises(ValueError):
        dataset[(999, 999)]
    readers[0].terminal = False
    with pytest.raises(ValueError):
        dataset[(999, 1000)]


def test_native_fusion_passes_real_state_and_preserves_unclipped_mean():
    class MeanPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def get_action(self, inputs):
            self.seen.append({key: value.clone() for key, value in inputs.items()})
            return torch.tensor([[2., -.5, 0., .25, -2.]])

    model = MeanPolicy()
    image = np.zeros((224, 224, 3), np.uint8)
    agent = SingleFrameAgent(model, lambda: image,
        lambda rgb: torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float(), state_fusion=True)
    agent.set_goal_rgb(image)
    current, goal = np.arange(28, dtype=np.float32), np.arange(28, dtype=np.float32) + 100
    action = agent.sample_actions(current, goal)
    assert action.shape == (5,) and action.dtype == np.float32
    np.testing.assert_array_equal(action, [2., -.5, 0., .25, -2.])
    torch.testing.assert_close(model.seen[0]['current_state'], torch.from_numpy(current)[None], rtol=0, atol=0)
    torch.testing.assert_close(model.seen[0]['goal_state'], torch.from_numpy(goal)[None], rtol=0, atol=0)
    assert agent.model_calls == 1 and agent.policy_warmup_steps == 0
    with pytest.raises(ValueError):
        agent.sample_actions(current[:3], goal)
