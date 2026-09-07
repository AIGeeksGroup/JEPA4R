"""Verify the actual encoder dependency boundary on CPU, without a simulator."""

import pytest
import torch

from ms2_lewm.model import WorldModel
from ms2_lewm.rl_components import OfficialPointNetFrameEncoder


@pytest.mark.parametrize("point_format", ["pointcloud", "xyz_rgb"])
def test_world_encoder_uses_only_xyz_rgb_history(point_format):
    torch.manual_seed(7319)
    model = WorldModel(action_dim=7).cpu().eval()
    points = torch.randn(2, model.history_size, 24, 6)
    if point_format == "pointcloud":
        info = {"pointcloud": points}
    else:
        info = {"xyz": points[..., :3], "rgb": points[..., 3:]}
    # These fields are useful to collectors/predictors but may never leak into
    # latent1 or latent2. Include the same action shape used in real training.
    info.update(
        state=torch.randn(2, model.history_size, 30),
        action=torch.randn(2, model.history_size, 7),
        goal=torch.randn(2, 3),
        sim_state=torch.randn(2, model.history_size, 64),
    )
    with torch.no_grad():
        reference = model.encode(info)
        changed = dict(info)
        for key in ("state", "action", "goal", "sim_state"):
            changed[key] = info[key] * -100.0 + 31.0
        alternate = model.encode(changed)
        points_only = model.encode(points)
        changed_points = points.clone()
        changed_points[..., 3:] *= -3.0
        changed_cloud = model.encode(changed_points)
    assert model.frame_encoder.input_channels == 6
    assert set(model.frame_encoder.obs_shape) == {"xyz", "rgb"}
    for key in ("latent1", "latent2"):
        torch.testing.assert_close(reference[key], alternate[key], rtol=0, atol=0)
        torch.testing.assert_close(reference[key], points_only[key], rtol=0, atol=0)
        assert not torch.equal(reference[key], changed_cloud[key])
    assert not torch.equal(reference["act_emb"], alternate["act_emb"])
    assert reference["latent1"].shape == (2, 3, 512)
    assert reference["latent2"].shape == (2, 512)


def test_shared_frame_encoder_ignores_non_point_observation_fields():
    torch.manual_seed(7320)
    frame = OfficialPointNetFrameEncoder().cpu().eval()
    points = {"xyz": torch.randn(2, 24, 3), "rgb": torch.rand(2, 24, 3)}
    full_obs = dict(points, state=torch.randn(2, 30), action=torch.randn(2, 7),
                    goal=torch.randn(2, 3), sim_state=torch.randn(2, 64))
    with torch.no_grad():
        actual = frame(full_obs)
        expected = frame(points)
        # The underlying PPO visual encoder obeys the same boundary.
        raw_visual = frame.pointnet(full_obs)
    assert actual.shape == (2, 512)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, raw_visual, rtol=0, atol=0)
