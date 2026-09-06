"""Newly generated synthetic fixtures shared by the CPU compatibility checks."""

import pytest


@pytest.fixture(scope="session")
def source_state():
    # One update/file per test session, including when either stage-three test
    # module is run alone. No prerequisite checkpoint is shipped or downloaded.
    from test_frozen_ppo import build_source_state

    return build_source_state()
