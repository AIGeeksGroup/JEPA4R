"""Small GPU GCIQL fixture test; preserves official sampler/reward/update math."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
IMPL = ROOT/'source/ogbench/impls'
sys.path.insert(0, str(IMPL))
import flax
import jax
import ogbench
from utils.datasets import Dataset, GCDataset
from utils.flax_utils import save_agent, restore_agent
from cached_inputs import CachedInputs
from cube_inputs import selftest

ADAPTED = ROOT/'reference/downstream/cached_lewm_visual_state_gciql_b1024_1m_109_20261013_v1/core_v1/gciql.py'
spec = importlib.util.spec_from_file_location('reused_optional_goal_gciql', ADAPTED)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def config():
    cfg = module.get_config()
    cfg.alpha = 1.0
    return cfg


def fingerprint(agent):
    h = hashlib.sha256()
    leaves = jax.tree_util.tree_leaves(flax.serialization.to_state_dict(agent))
    for value in leaves:
        a = np.asarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return dict(sha256=h.hexdigest(), leaves=len(leaves))


def metrics(values):
    result = {k: float(v) for k,v in values.items()}
    assert all(np.isfinite(v) for v in result.values())
    return result


def factory(batch):
    return module.GCIQLAgent.create(0, batch['observations'][:1], batch['actions'][:1],
                                   config(), ex_goals=batch['actor_goals'][:1])


def main(fresh=False):
    assert jax.default_backend() == 'gpu'
    if fresh:
        b = dict(np.load(ROOT/'smoke/reload_batch_v4.npz'))
        agent = restore_agent(factory(b), str(ROOT/'smoke/gciql_v4'), 2)
        actual = fingerprint(agent)
        expected = json.loads((ROOT/'smoke/gciql_v4_report.json').read_text())['saved_state']
        assert actual == expected
        actions = np.asarray(agent.sample_actions(b['observations'], b['actor_goals'],
                             seed=jax.random.PRNGKey(0), temperature=0))
        assert actions.shape == (16,5) and np.isfinite(actions).all()
        (ROOT/'smoke/reload_report_v4.json').write_text(json.dumps(dict(actual_exit_code=0,
            full_state_exact=actual, finite_actions=True, completed=time.time()), indent=2))
        return
    original_tree = ast.parse((IMPL/'agents/gciql.py').read_text())
    adapted_tree = ast.parse(ADAPTED.read_text())
    original = {n.name:ast.dump(n,include_attributes=False) for node in original_tree.body
                if isinstance(node,ast.ClassDef) for n in node.body if isinstance(n,ast.FunctionDef)}
    adapted = {n.name:ast.dump(n,include_attributes=False) for node in adapted_tree.body
               if isinstance(node,ast.ClassDef) for n in node.body if isinstance(n,ast.FunctionDef)}
    assert all(adapted[name]==value for name,value in original.items() if name!='create')
    fixture = dict(np.load(ROOT/'smoke/features.npz'))
    terminals = np.zeros(8,np.float32); terminals[-1] = 1
    fixture_path = ROOT/'smoke/fixture_dataset_v4.npz'
    np.savez(fixture_path, observations=fixture['observations'], actions=fixture['actions'], terminals=terminals)
    raw = ogbench.load_dataset(str(fixture_path), compact_dataset=True)
    dataset = Dataset.create(**raw)
    cfg = config()
    original = GCDataset(dataset,cfg)
    adapted = CachedInputs(dataset,cfg,features=fixture['features_2d'],cubes=3)
    np.random.seed(0); expected = original.sample(16); state_a = np.random.get_state()
    np.random.seed(0); batch = adapted.sample(16); state_b = np.random.get_state()
    for key in ['actions','rewards','masks']:
        np.testing.assert_array_equal(batch[key],expected[key])
    np.testing.assert_array_equal(state_a[1],state_b[1]);assert state_a[2:]==state_b[2:]
    np.testing.assert_array_equal(batch['observations'][:,192:],expected['observations'][:,:19])
    np.testing.assert_array_equal(batch['observations'][:,:192],fixture['features_2d'][adapted.last_ids['current']])
    for key,idskey in [('value_goals','value_goal'),('actor_goals','actor_goal')]:
        xyz=np.concatenate([expected[key][:,19+9*i:22+9*i] for i in range(3)],axis=-1)
        np.testing.assert_array_equal(batch[key][:,192:],xyz)
        np.testing.assert_array_equal(batch[key][:,:192],fixture['features_2d'][adapted.last_ids[idskey]])
    agent = factory(batch)
    losses=[]
    for _ in range(2):
        batch=adapted.sample(16);agent,info=agent.update(batch)
        values=metrics(info);assert values['grad/norm']>0;losses.append(values)
    assert int(agent.network.step)==3 and int(agent.network.opt_state[0].count)==2
    validation=metrics(agent.total_loss(adapted.sample(16),grad_params=None)[1])
    # Also validate the PCD cache interface, without constructing a second policy.
    pcd=CachedInputs(dataset,cfg,features=fixture['features_3d'],cubes=3).sample(16)
    assert pcd['observations'].shape==(16,211) and pcd['actor_goals'].shape==(16,201)
    saved=fingerprint(agent)
    (ROOT/'smoke/gciql_v4').mkdir(exist_ok=False)
    save_agent(agent,str(ROOT/'smoke/gciql_v4'),2)
    np.savez(ROOT/'smoke/reload_batch_v4.npz',**batch)
    report=dict(actual_exit_code=0,completed=time.time(),official_task_alpha=1.0,
        official_config=cfg.to_dict(),official_update_methods_AST_exact=True,
        reused_goal_factory_interface=True,official_sampler_reward_and_host_RNG_equal=True,
        fixture_only=True,fixture_cut_not_true_terminal=True,actual_smoke_batch_size=16,
        formal_batch_size=1024,formal_training_started=False,SR_started=False,
        two_update_metrics=losses,validation=validation,saved_state=saved,
        network_step=3,Adam_count=2,interface_tests=selftest(),
        goal_dims=201,current_dims=211,renderer_or_encoder_calls_during_updates=0,
        JAX=jax.__version__,JAX_default_precision=str(jax.config.jax_default_matmul_precision))
    (ROOT/'smoke/gciql_v4_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main('--fresh' in sys.argv)
