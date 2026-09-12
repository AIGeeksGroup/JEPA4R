from types import SimpleNamespace

import numpy as np
import pytest

from scripts.generate_liftcube_contact_sidecar import (
    EpisodeContactRecord,
    alignment_errors,
    assemble_sidecar,
    cube_robot_contact_from_contacts,
    frame_ids_for_actions,
    validate_alignment,
)


def _actor(name):
    return SimpleNamespace(name=name)


def _contact(left, right):
    return SimpleNamespace(actor0=_actor(left), actor1=_actor(right))


def test_cube_robot_contact_excludes_table_and_accepts_robot_link():
    contacts = [
        _contact("cube", "ground"),
        _contact("cube", "panda_leftfinger"),
    ]

    assert cube_robot_contact_from_contacts(
        contacts, robot_actor_names={"panda_leftfinger"}
    )
    assert not cube_robot_contact_from_contacts(
        [_contact("cube", "ground")], robot_actor_names={"panda_leftfinger"}
    )


def test_frame_ids_include_reset_and_every_post_action_observation():
    actions = np.zeros((3, 7), dtype=np.float32)

    np.testing.assert_array_equal(frame_ids_for_actions(actions), [0, 1, 2, 3])


def test_alignment_errors_are_per_frame_euclidean_distances():
    reference_object = np.zeros((3, 3), dtype=np.float64)
    replay_object = reference_object.copy()
    replay_object[1, 0] = 3e-5
    reference_tcp = np.ones((3, 3), dtype=np.float64)
    replay_tcp = reference_tcp.copy()
    replay_tcp[2, 1] += 4e-5

    object_error, tcp_error = alignment_errors(
        replay_object, reference_object, replay_tcp, reference_tcp
    )

    np.testing.assert_allclose(object_error, [0.0, 3e-5, 0.0])
    np.testing.assert_allclose(tcp_error, [0.0, 0.0, 4e-5])
    validate_alignment(object_error, tcp_error, tolerance_m=5e-5)
    with pytest.raises(ValueError, match="alignment"):
        validate_alignment(object_error, tcp_error, tolerance_m=2e-5)


def test_assemble_sidecar_sorts_episode_frame_rows_and_preserves_schema():
    records = [
        EpisodeContactRecord(
            episode_id=2,
            frame_id=np.asarray([0, 1]),
            cube_robot_contact=np.asarray([False, True]),
            object_pos_error_m=np.asarray([0.0, 1e-6]),
            tcp_pos_error_m=np.asarray([0.0, 2e-6]),
        ),
        EpisodeContactRecord(
            episode_id=1,
            frame_id=np.asarray([0, 1]),
            cube_robot_contact=np.asarray([False, False]),
            object_pos_error_m=np.asarray([3e-6, 4e-6]),
            tcp_pos_error_m=np.asarray([5e-6, 6e-6]),
        ),
    ]

    sidecar = assemble_sidecar(records)

    assert set(sidecar) == {
        "episode_id",
        "frame_id",
        "cube_robot_contact",
        "object_pos_error_m",
        "tcp_pos_error_m",
    }
    np.testing.assert_array_equal(sidecar["episode_id"], [1, 1, 2, 2])
    np.testing.assert_array_equal(sidecar["frame_id"], [0, 1, 0, 1])
    np.testing.assert_array_equal(sidecar["cube_robot_contact"], [0, 0, 0, 1])


def test_assemble_sidecar_rejects_duplicate_episode_frame_rows():
    record = EpisodeContactRecord(
        episode_id=0,
        frame_id=np.asarray([1, 1]),
        cube_robot_contact=np.asarray([False, True]),
        object_pos_error_m=np.zeros(2),
        tcp_pos_error_m=np.zeros(2),
    )

    with pytest.raises(ValueError, match="duplicate"):
        assemble_sidecar([record])
