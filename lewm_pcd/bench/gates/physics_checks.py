"""Bounded physical TimeLimit/goal interfaces; no encoder or policy benchmark."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source_root.resolve()
    manifest_path = source / 'SOURCE_MANIFEST.json'
    if sha(manifest_path) != args.source_manifest_sha256:
        raise ValueError('physical fixture source manifest differs')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    paths = ['lewm_pcd/third_party/ogbench/ogbench/manipspace/__init__.py',
             'lewm_pcd/third_party/ogbench/ogbench/manipspace/envs/cube_env.py',
             'lewm_pcd/third_party/ogbench/ogbench/manipspace/envs/manipspace_env.py',
             'lewm_pcd/third_party/ogbench/impls/utils/evaluation.py',
             'lewm_pcd/bench/ogbench/downstream/vendor/physics.py',
             'lewm_pcd/bench/ogbench/downstream/vendor/cube_inputs.py']
    for path in paths:
        if sha(source / path) != manifest['files'][path]:
            raise ValueError('physical fixture runtime source differs: ' + path)
    args.output.mkdir(parents=True, exist_ok=False)
    lewm = source / 'lewm_pcd'
    sys.path[:0] = [str(lewm / 'third_party/ogbench'), str(lewm),
                   str(lewm / 'third_party/stable-worldmodel')]
    import numpy as np
    import gymnasium as gym
    import ogbench
    import jax
    if jax.default_backend() != 'cpu':
        raise ValueError('physical boundary fixture requires the CPU JAX backend')
    expected_ogbench = lewm / 'third_party/ogbench/ogbench/__init__.py'
    if Path(ogbench.__file__).resolve() != expected_ogbench.resolve():
        raise ValueError('physical fixture imported another OGBench')
    evaluator = load('_gates_official_evaluator', source / paths[3])
    physics = load('_gates_original_numpy_runtime', source / paths[4])
    seam = load('_gates_cube_state_projection', source / paths[5])
    from bench.ogbench.workflow_downstream import group_contract
    from bench.ogbench.collection import build_plan
    runtime = physics.configure_numpy_physics_runtime()

    class ZeroActionFixture:
        def sample_actions(self, **kwargs):
            return np.zeros(5, dtype=np.float32)

    class NearTimeLimit(gym.Wrapper):
        """Expose only the final two counter transitions, keeping real physics."""
        def reset(self, *, seed=None, options=None):
            observation, info = self.env.reset(seed=970100000, options=options)
            self.env._elapsed_steps = self.env._max_episode_steps - 2
            self.transitions = []
            self.after_terminal_calls = 0
            self.ended = False
            self.initial = (observation.copy(), np.asarray(info['goal']).copy())
            return observation, info

        def step(self, action):
            if self.ended:
                self.after_terminal_calls += 1
                raise AssertionError('official evaluator stepped after the terminal boundary')
            before = self.env._elapsed_steps
            result = self.env.step(action)
            self.transitions.append(dict(before=before, after=self.env._elapsed_steps,
                                         terminated=bool(result[2]), truncated=bool(result[3])))
            self.ended = bool(result[2] or result[3])
            return result

    cases = []
    for count, name, horizon in [(1, 'single', 200), (2, 'double', 500),
                                  (3, 'triple', 1000), (4, 'quadruple', 1000)]:
        contract = group_contract('lewm_pcd', count)
        assert contract['native_max_episode_steps'] == horizon
        collection = build_plan(count, args.output / ('uncreated_collection_' + name), mode='smoke')
        env = NearTimeLimit(gym.make('cube-' + name + '-v0', ob_type='states',
                                    render_mode=None, visualize_info=False, pixel_transparent_arm=False))
        try:
            assert env.action_space.shape == (5,) and env.env._max_episode_steps == horizon
            assert env.unwrapped._num_cubes == count and len(env.unwrapped.task_infos) == 5
            evaluator.evaluate(ZeroActionFixture(), env, task_id=1, config={'discrete': False},
                               num_eval_episodes=1, num_video_episodes=0, eval_temperature=0)
            assert len(env.transitions) == 2 and env.after_terminal_calls == 0
            assert env.transitions[0]['after'] == horizon - 1 and not env.transitions[0]['truncated']
            assert env.transitions[1]['after'] == horizon and env.transitions[1]['truncated']
            current, goal = env.initial
            projection = seam.CubeInputs(count)
            placeholder = np.zeros((1, 192), dtype=np.float32)
            x = projection.current(placeholder, current.astype(np.float32)[None])
            g = projection.goal(placeholder, goal.astype(np.float32)[None])
            assert x.shape == (1, 211) and g.shape == (1, 192 + 3 * count)
            np.savez(args.output / (name + '_state_inputs.npz'), state=current, goal_state=goal,
                     projected_current=x, projected_goal=g)
            cases.append(dict(cubes=count, native_horizon=horizon, collection_horizon=collection['raw_horizon'],
                              fixture_initial_elapsed_steps=horizon-2, actual_physics_steps=2,
                              after_terminal_step_calls=0, transitions=env.transitions,
                              state_dim=int(current.size), goal_input_dim=int(g.shape[1])))
        finally:
            env.close()
    report = dict(actual_exit_code=0, cases=cases, numpy_physics_runtime=runtime,
                  source_manifest_sha256=args.source_manifest_sha256,
                  source_files={path: sha(source / path) for path in paths}, producer_sha256=sha(__file__),
                  scope='Actual four-environment final-two-TimeLimit-counter fixtures with original evaluator. '
                        'Zero-action agent and zero latents are interface placeholders only; '
                        'no full-horizon rollout, encoder forward, learned control, training, rendering or success-rate result.')
    (args.output / 'RESULT.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    (args.output / 'actual_exit_code').write_text('0\n')
    print(json.dumps({'actual_exit_code': 0, 'cases': cases}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
