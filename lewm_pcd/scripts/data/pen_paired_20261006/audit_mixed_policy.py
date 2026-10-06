"""Offline frozen-policy recomputation from sealed mixed Pen data.

No native environment is constructed or stepped. RecordedNativeObs exposes
only a stored45D observation. The unchanged production feedback policy owns
mean/noise/random/clip behavior; wrappers record its actual draws for audit.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace as NS

import numpy as np

EXPERT_SHA = '061dba4907a76b83139ae5a8d0809faa9774dd028d12ce869036b620bd728740'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda:file.read(1 << 20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


class RecordedNativeObs:
    def __init__(self, observations):
        self.observations,self.index = observations,0
    def get_obs(self):
        return self.observations[self.index].copy()


class DrawRecorder:
    def __init__(self, generator):
        self.generator = generator
        self.normal_draws,self.uniform_draws = [],[]
    @property
    def bit_generator(self):
        return self.generator.bit_generator
    def normal(self,*args,**kwargs):
        value = self.generator.normal(*args,**kwargs)
        self.normal_draws.append(np.asarray(value).copy())
        return value
    def uniform(self,*args,**kwargs):
        value = self.generator.uniform(*args,**kwargs)
        self.uniform_draws.append(np.asarray(value).copy())
        return value


def recompute_episode(policy, data, metadata, action_atol=0.):
    n = len(data['action'])
    assert 2 <= n <= 101
    assert data['action'].shape == (n,24) and data['action'].dtype == np.float32
    assert data['native_observation'].shape == (n,45) and np.isfinite(data['native_observation']).all()
    np.testing.assert_array_equal(data['observation'],data['native_observation'])
    assert np.isnan(data['action'][-1]).all() and np.isfinite(data['action'][:-1]).all()
    np.testing.assert_array_equal(data['executed_action'][1:],data['action'][:-1])
    assert not data['executed_action'][0].any()
    np.testing.assert_array_equal(data['executed_action_valid'],np.r_[0,np.ones(n-1,dtype=np.int64)])
    np.testing.assert_array_equal(data['step_idx'],np.arange(n))
    np.testing.assert_array_equal(data['native_step_idx'],np.arange(n))
    assert np.all(data['episode_seed'] == metadata['seed'])
    kind,sigma,start,length = (metadata[key] for key in ('sample_type','noise_std','branch_t','branch_len'))
    kind,start,length = int(kind),int(start),int(length)
    sigma = float(sigma)
    assert kind in (0,1,2) and 0 <= sigma < .2
    for key,value in (('sample_type',kind),('noise_std',sigma),('branch_t',start),('branch_len',length)):
        assert np.all(data[key] == value), 'episode plan mismatch: '+key
    if kind == 1:
        assert 15 <= start < 60 and (10 <= length <= 30 or length == 100)
        expected_random = (np.arange(n) >= start) & (np.arange(n) < start+length)
    else:
        assert start == -1 and length == 0
        expected_random = np.full(n,kind == 2)
    if kind == 2:
        assert sigma == 0
    np.testing.assert_array_equal(data['rand_action'].astype(bool),expected_random)
    for key in ('before','after'):
        if 'policy_rng_'+key+'_sha256' in metadata:
            assert value_sha(metadata['policy_rng_'+key]) == metadata['policy_rng_'+key+'_sha256']
    native = RecordedNativeObs(data['native_observation'])
    inner = NS(env_name='pen-v0',action_space=NS(shape=(24,)),native_env=native)
    policy.set_env(NS(envs=[NS(unwrapped=inner)],num_envs=1))
    generator = policy.rng.generator if isinstance(policy.rng,DrawRecorder) else policy.rng
    generator.bit_generator.state = copy.deepcopy(metadata['policy_rng_before'])
    policy.rng = DrawRecorder(generator)
    expected = np.empty((n-1,24),np.float32)
    for step in range(n-1):
        native.index = step
        infos = {'rand_action':np.asarray([[data['rand_action'][step]]]),
                 'noise_std':np.asarray([[data['noise_std'][step]]])}
        expected[step] = policy.get_action(infos)[0]
    difference = float(np.max(np.abs(expected.astype(np.float64)-data['action'][:-1])))
    assert difference <= action_atol, 'actual action differs from original frozen policy/RNG; max_abs='+str(difference)
    assert value_sha(policy.rng.bit_generator.state) == value_sha(metadata['policy_rng_after']), 'end policy RNG state mismatch'
    random_steps = int(np.count_nonzero(expected_random[:-1]))
    expert_steps = n-1-random_steps
    assert len(policy.rng.normal_draws) == expert_steps and len(policy.rng.uniform_draws) == random_steps
    assert all(draw.shape == (24,) and np.isfinite(draw).all() for draw in policy.rng.normal_draws+policy.rng.uniform_draws)
    assert all((np.abs(draw) <= 1.).all() for draw in policy.rng.uniform_draws)
    return {'sampling_type':kind,'actor_steps':n-1,'expert_steps':expert_steps,'random_steps':random_steps,
            'noise_std':sigma,'branch_t':start,'branch_len':length,
            'nonzero_gaussian_draw_steps':sum(bool(np.any(draw)) for draw in policy.rng.normal_draws),
            'gaussian_draw_abs_max':max((float(np.max(np.abs(draw))) for draw in policy.rng.normal_draws),default=0.),
            'action_max_abs_error':difference,'all_actions_bitwise_equal':bool(np.array_equal(expected,data['action'][:-1])),
            'end_rng_exact':True,'terminal_action_all_nan':True}


def load_file(path,name):
    spec = importlib.util.spec_from_file_location(name,str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args):
    if args.out.exists():
        raise FileExistsError('fresh offline audit output directory required')
    args.out.mkdir(parents=True)
    report = {'started_at':datetime.now(timezone.utc).isoformat(),'episodes':[],'errors':[],
              'scope':'offline_original_policy_and_episode_plan_recompute_no_native_rollouts_no_training',
              'native_environments_constructed':0,'simulation_steps':0,'status':'UNVERIFIED','exit_code':1,
              'action_atol':args.action_atol,'source_sha256':{},'input_sha256':{}}
    try:
        assert 0 <= args.action_atol <= 1e-6
        if args.mjrl_root is not None:
            sys.path.insert(0,str(args.mjrl_root.resolve()))
        import torch
        torch.set_num_threads(1)
        policy_path = args.code_root/'lewm_pcd/envs/adroit/policy.py'
        env_path = args.code_root/'lewm_pcd/envs/adroit/env.py'
        report['source_sha256'] = {str(path):sha(path) for path in (policy_path,env_path,args.expert_path,Path(__file__))}
        assert sha(args.expert_path) == EXPERT_SHA
        policy_module = load_file(policy_path,'offline_original_pen_policy')
        env_module = load_file(env_path,'offline_original_pen_plan')
        policy = policy_module.AdroitFeedbackPolicy(args.expert_path,EXPERT_SHA,env_name='pen-v0',seed=0)
        plan_self = NS(force_type=None,mix=np.array([.5,.3,.2]),noise_max=.2,episode_len=100,
                       branch_tail_prob=.5,burst_len=(10,30))
        manifest = json.loads(args.manifest.read_text())
        report['input_sha256'][str(args.manifest)] = sha(args.manifest)
        for record in manifest['episodes']:
            path,meta_path = Path(record['rgb']),Path(record['metadata'])
            metadata = json.loads(meta_path.read_text())
            report['input_sha256'].update({str(path):sha(path),str(meta_path):sha(meta_path)})
            expected_payload = metadata.get('rgb_payload_sha256',metadata.get('payload_sha256'))
            if path.suffix == '.npz':
                assert sha(path) == expected_payload, 'sealed RGB payload hash mismatch'
                with np.load(str(path),allow_pickle=False) as archive:
                    data = {key:archive[key] for key in archive.files}
            else:
                assert path.suffix == '.h5' and 'episode_index' in record
                import h5py
                with h5py.File(path,'r') as archive:
                    index = int(record['episode_index']);start = int(archive['ep_offset'][index]);end = start+int(archive['ep_len'][index])
                    data = {key:archive[key][start:end] for key in archive if key not in ('ep_len','ep_offset')}
            assert metadata['source_hashes']['production_env'] == sha(env_path)
            assert metadata['source_hashes']['expert_adapter'] == sha(policy_path)
            kind,sigma,start,length = env_module.AdroitEnv._draw_plan(plan_self,np.random.default_rng(int(metadata['seed'])))
            assert kind == int(metadata['sample_type']) and sigma == float(metadata['noise_std'])
            assert start == int(metadata['branch_t']) and length == int(metadata['branch_len']), 'episode plan differs from original seed/formula'
            with torch.no_grad():
                result = dict(seed=int(metadata['seed']),episode_id=int(metadata['episode_id']),
                              **recompute_episode(policy,data,metadata,args.action_atol))
            report['episodes'].append(result)
            print(json.dumps(result),flush=True)
        kinds = {row['sampling_type'] for row in report['episodes']}
        assert kinds == {0,1,2}, 'real gate must cover all three episode types'
        assert all(any(row['sampling_type'] == kind and row['nonzero_gaussian_draw_steps'] > 0 for row in report['episodes']) for kind in (0,1)), 'missing real Gaussian action coverage'
        assert any(row['sampling_type'] == 1 and row['expert_steps'] > 0 and row['random_steps'] > 0 for row in report['episodes']), 'missing real branch transition coverage'
        assert all(sha(path) == digest for path,digest in report['input_sha256'].items()), 'audit changed input bytes'
        assert all(sha(path) == digest for path,digest in report['source_sha256'].items()), 'frozen source changed during audit'
        report.update(status='PASS',exit_code=0,coverage_types=[0,1,2],real_checkpoint=True,
                      versions={'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
                                'gymnasium':importlib.metadata.version('gymnasium')})
    except BaseException:
        report.update(status='FAIL',exit_code=1)
        report['errors'].append(traceback.format_exc())
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        (args.out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
        (args.out/'exit_code').write_text(str(report['exit_code'])+'\n',encoding='utf-8')
    return report['exit_code']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('code-root','expert-path','manifest','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--mjrl-root',type=Path)
    parser.add_argument('--action-atol',type=float,default=0.)
    raise SystemExit(run(parser.parse_args()))
