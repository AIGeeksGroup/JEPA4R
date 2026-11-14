"""CPU unit checks of the deterministic action reference that the GCIQL reload check uses."""

import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import json
import random
import hashlib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def boundary_api():
    path = Path(__file__).resolve().parents[2] / "jepa4r/policy/gciql/downstream/vendor/boundary.py"
    fake = ModuleType("trainer_runtime")
    for name in (
        "agent_config",
        "core",
        "create_agent",
        "fingerprint",
        "get_batch",
        "initial_agent",
        "jax",
        "model",
        "np",
        "prepare",
        "restore_agent",
        "torch",
        "utility",
    ):
        setattr(fake, name, None)
    fake.np = np
    spec = importlib.util.spec_from_file_location("_gciql_action_reference_unit", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"trainer_runtime": fake}):
        spec.loader.exec_module(module)
    return module


class ReloadActionTests(unittest.TestCase):
    def setUp(self):
        self.api = boundary_api()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "actions.npy"
        self.reference = np.arange(1024 * 5, dtype=np.float32).reshape(1024, 5) / 10000
        np.save(self.path, self.reference)

    def test_full_reference_arrays_pass_and_exact_error_count_is_reported(self):
        result = self.api.compare_reload_actions(self.reference.copy(), self.path)
        self.assertTrue(result["passed"])
        self.assertEqual(result["out_of_tolerance"], 0)
        self.assertEqual(result["max_abs_error"], 0.0)

    def test_one_non_sampled_action_element_is_detected(self):
        actual = self.reference.copy()
        actual[731, 4] += 0.01
        result = self.api.compare_reload_actions(actual, self.path)
        self.assertFalse(result["passed"])
        self.assertEqual(result["out_of_tolerance"], 1)

    def test_reference_forward_keeps_agent_and_host_rng_unchanged(self):
        api = self.api
        api.jax = SimpleNamespace(random=SimpleNamespace(PRNGKey=lambda seed: ("key", seed)))
        host = dict(python=(1, 2), numpy=np.arange(4), torch=np.arange(4, dtype=np.uint8), cuda=[])
        api.core = SimpleNamespace(host_rng=lambda: host)
        api.fingerprint = lambda agent: {"all_state": agent.state}

        class Agent:
            state = 7

            def sample_actions(inner, observations, goals, seed, temperature):
                self.assertEqual(seed, ("key", 0))
                self.assertEqual(temperature, 0)
                return self.reference.copy()

        batch = dict(observations=np.ones((1024, 211), np.float32), actor_goals=np.ones((1024, 195), np.float32))
        value, proof = api.deterministic_action_reference(Agent(), batch)
        np.testing.assert_array_equal(value, self.reference)
        self.assertTrue(proof["full_agent_state_unchanged"])
        self.assertTrue(proof["host_rng_unchanged"])

    def test_reference_forward_rejects_mutating_agent_or_rng(self):
        api = self.api
        api.jax = SimpleNamespace(random=SimpleNamespace(PRNGKey=lambda seed: ("key", seed)))
        host = dict(python=(1, 2), numpy=np.arange(4), torch=np.arange(4, dtype=np.uint8), cuda=[])
        api.core = SimpleNamespace(host_rng=lambda: host)
        api.fingerprint = lambda agent: {"all_state": agent.state}
        batch = dict(observations=np.ones((1024, 211), np.float32), actor_goals=np.ones((1024, 195), np.float32))

        class MutatingAgent:
            state = 7

            def sample_actions(inner, *args, **kwargs):
                inner.state += 1
                return self.reference.copy()

        with self.assertRaises(ValueError):
            api.deterministic_action_reference(MutatingAgent(), batch)

        class MutatingRng:
            state = 7

            def sample_actions(inner, *args, **kwargs):
                host["numpy"][1] += 1
                return self.reference.copy()

        with self.assertRaises(ValueError):
            api.deterministic_action_reference(MutatingRng(), batch)

    def test_params_only_fingerprint_excludes_counter_and_includes_weights(self):
        import hashlib

        api = self.api
        fake = ModuleType("flax")
        fake.serialization = SimpleNamespace(to_state_dict=lambda agent: agent)
        api.core = SimpleNamespace(
            digest_array=lambda a: hashlib.sha256(
                str(np.asarray(a).dtype).encode() + str(np.asarray(a).shape).encode() + np.asarray(a).tobytes()
            ).hexdigest()
        )
        first = dict(network=dict(params={"weight": np.arange(6, dtype=np.float32)}, step=1))
        second = dict(network=dict(params={"weight": np.arange(6, dtype=np.float32)}, step=2))
        with patch.dict(sys.modules, {"flax": fake}):
            self.assertEqual(api.parameter_fingerprint(first), api.parameter_fingerprint(second))
            second["network"]["params"]["weight"][4] += 0.1
            self.assertNotEqual(api.parameter_fingerprint(first), api.parameter_fingerprint(second))

    def reload_fixture(self):
        api = self.api
        folder = Path(self.tmp.name) / "gate"
        folder.mkdir()
        batch = dict(observations=np.ones((1024, 211), np.float32), actor_goals=np.ones((1024, 195), np.float32))
        ids = {"current": np.arange(1024, dtype=np.int64)}
        np.savez(folder / "next_batch.npz", **batch)
        np.savez(folder / "next_ids.npz", **ids)
        np.save(folder / "next_actions.npy", self.reference)
        fp = {"leaves": 153, "sha256": "fixture"}

        def sha(p):
            return hashlib.sha256(Path(p).read_bytes()).hexdigest()

        (folder / "report.json").write_text(
            json.dumps(
                dict(
                    actual_exit_code=0,
                    role="unit",
                    updates=2,
                    saved_state=fp,
                    action_reference_sha256=sha(folder / "next_actions.npy"),
                )
            )
        )

        class Agent:
            def sample_actions(inner, *args, **kwargs):
                return self.reference + 0.01

        agent = Agent()
        samplers = {"train": SimpleNamespace(last_ids=ids)}
        sidecar = dict(
            binding={"source_bindings": {"unit": True}},
            rng=dict(numpy=np.random.get_state(), python=random.getstate(), torch=np.zeros(3, np.uint8)),
        )
        api.jax = SimpleNamespace(default_backend=lambda: "cpu", random=SimpleNamespace(PRNGKey=lambda key: key))
        api.prepare = lambda cfg: (samplers, SimpleNamespace(close=lambda: None), {"source_bindings": {"unit": True}})
        api.initial_agent = lambda *a: (agent, batch)
        api.create_agent = lambda *a: agent
        api.model = SimpleNamespace(GCIQLAgent=object)
        api.agent_config = lambda cfg: {}
        api.restore_saved = lambda *a: (folder, sidecar)
        api.restore_agent = lambda *a: agent
        api.fingerprint = lambda *a: fp
        api.get_batch = lambda *a: batch
        api.torch = SimpleNamespace(set_rng_state=lambda value: None)
        api.core = SimpleNamespace(
            host_rng=lambda: dict(python=(1, 2), numpy=np.arange(4), torch=np.arange(4, dtype=np.uint8), cuda=[])
        )
        api.utility = SimpleNamespace(sha=sha, atomic_json=lambda p, v: Path(p).write_text(json.dumps(v)))
        return folder, {"role": "unit", "seed": 0}

    def test_legacy_reload_keeps_compatibility_and_labels_cross_backend_diagnostic(self):
        folder, cfg = self.reload_fixture()
        report = Path(self.tmp.name) / "legacy_reload.json"
        self.api.reload_gate(cfg, folder, report)
        result = json.loads(report.read_text())
        self.assertEqual(result["actual_exit_code"], 0)
        self.assertFalse(result["cross_backend_diagnostic"]["passed"])
        self.assertFalse(result["same_cpu_action_consistency_verified"])

    def test_guide_reload_requires_an_independent_cpu_reference(self):
        folder, cfg = self.reload_fixture()
        with self.assertRaises(ValueError):
            self.api.reload_gate(cfg, folder, Path(self.tmp.name) / "guide.json", require_cpu_reference=True)


if __name__ == "__main__":
    unittest.main()
