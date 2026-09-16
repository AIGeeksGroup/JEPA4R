from pathlib import Path
import sys

import numpy as np
import pytest


DATA_SCRIPTS = Path(__file__).resolve().parents[1] / "data"
sys.path.insert(0, str(DATA_SCRIPTS))

from maniskill_pickcube import validate_episode_arrays  # noqa: E402


def _arrays(group=None):
    arrays = {
        "rgb": np.zeros((2, 8, 8, 3), dtype=np.uint8),
        "depth": np.zeros((2, 8, 8, 1), dtype=np.int16),
        "pcd_xyz": np.zeros((2, 12, 3), dtype=np.float32),
        "pcd_rgb": np.zeros((2, 12, 3), dtype=np.uint8),
        "proprio": np.zeros((2, 25), dtype=np.float32),
        "action": np.vstack(
            [np.zeros((1, 8), dtype=np.float32), np.full((1, 8), np.nan, dtype=np.float32)]
        ),
    }
    if group is not None:
        arrays["pcd_group"] = group
    return arrays


def test_object_aware_group_contract_accepts_valid_labels():
    group = np.tile(np.array([1, 2, 3, 4], dtype=np.uint8), (2, 3))
    validate_episode_arrays(_arrays(group), expected_frames=2, num_points=12)


@pytest.mark.parametrize(
    "group,match",
    [
        (np.zeros((2, 11), dtype=np.uint8), "pcd_group shape"),
        (np.full((2, 12), 9, dtype=np.uint8), "pcd_group labels"),
        (np.zeros((2, 12), dtype=np.int32), "pcd_group dtype"),
    ],
)
def test_object_aware_group_contract_rejects_invalid_values(group, match):
    with pytest.raises(ValueError, match=match):
        validate_episode_arrays(_arrays(group), expected_frames=2, num_points=12)


def test_object_aware_group_contract_requires_cube_lower_bound():
    group = np.tile(np.array([2, 3, 4, 4], dtype=np.uint8), (2, 3))
    with pytest.raises(ValueError, match="Cube"):
        validate_episode_arrays(_arrays(group), expected_frames=2, num_points=12)
