"""State28 glue for the sealed original ReplayPairWrapper; no reset/physics implementation."""
import importlib.util
import json
from pathlib import Path
import numpy as np

LEGACY_SHA='5721760f53f8f3c4afb3f85626366022f844861ff3462a46bb5a0365f1e372bb'
REPLAY_SHA='fcbc9a13cc38b6ad6ad08bc61b8b463dbd35a76d6511c928e3a94a35c1d699f9'


def sealed_module(path,expected,name):
    import hashlib
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def source_modules(base):
    return (sealed_module(base/'official_state_sr.py',LEGACY_SHA,'sealed_original_state_sr'),
            sealed_module(base/'native_single_frame_adapter.py',REPLAY_SHA,'sealed_original_replay'))


def prepare_reference_replay(root,base,source):
    legacy,replay=source_modules(base)
    root=Path(root).resolve()
    records=replay.validate_references(root)
    baseline=json.loads((root/'protocol.json').read_text())
    source_seal=replay.seal_original_source(source,baseline)
    audit=json.loads((root/'independent_full_audit.json').read_text())
    assert audit['status']=='passed' and audit['exit_code']==0 and audit['episodes']==250
    manifest=json.loads((root/'episode_manifest.json').read_text())
    assert len(manifest)==250
    indexed={f"{row['task']}:{row['index']}":row for row in manifest}
    for row in records:
        old=indexed[f"{row['task']}:{row['index']}"]
        assert old['seed']==row['seed'] and old['reset_sha256']==row['reset_file_sha256']
    return {'reference_root':str(root),'reference_manifest':{f"{r['task']}:{r['index']}":r for r in records},
        'reference_protocol_sha256':legacy.sha(root/'protocol.json'),
        'reference_episode_manifest_sha256':legacy.sha(root/'episode_manifest.json'),
        'reference_independent_audit_sha256':legacy.sha(root/'independent_full_audit.json'),
        'legacy_helper_sha256':LEGACY_SHA,'replay_source_sha256':REPLAY_SHA,
        'source_seal':source_seal,'reset_pairing':'original ReplayPairWrapper: exact initial/goal/ctrl/mocap/private/RNG/model',
        'representation':'state28; no visual rendering or alpha exception'}


class StateAgent:
    """Preserve the original sample_actions call; supply the existing replay lifecycle API."""
    def __init__(self,agent,goal):
        self.agent=agent; self.expected_goal=np.array(goal,copy=True); self.reset()
    def reset(self):
        self.model_calls=0; self.goal=None
    def set_goal_rgb(self,state_goal):
        # The sealed wrapper calls this name; its payload here is a real state28 goal.
        state_goal=np.asarray(state_goal)
        assert state_goal.shape==(28,) and np.isfinite(state_goal).all()
        assert state_goal.dtype==self.expected_goal.dtype and np.array_equal(state_goal,self.expected_goal)
        self.goal=np.array(state_goal,copy=True)
    def sample_actions(self,observations,goals=None,seed=None,temperature=1.0):
        assert self.goal is not None and temperature==0
        assert np.asarray(observations).shape==(28,) and np.isfinite(observations).all()
        assert np.asarray(goals).dtype==self.goal.dtype and np.array_equal(goals,self.goal)
        answer=self.agent.sample_actions(observations=observations,goals=goals,seed=seed,temperature=temperature)
        self.model_calls+=1
        return answer


def make_replay_environment(env,agent,binding,task,index,attempt,base):
    import gymnasium as gym
    legacy,replay=source_modules(base)
    row=binding['reference_manifest'][f'{task}:{index}']
    path=Path(row['reset_path'])
    assert legacy.sha(path)==row['reset_file_sha256']
    reference=json.loads(path.read_text())
    record=reference['goal28']
    goal=np.asarray(record['values'],dtype=np.dtype(record['dtype']))
    actor=StateAgent(agent,goal)
    def on_reset(receipt):
        receipt=dict(receipt)
        receipt['goal_state28_bytes_sha256']=receipt.pop('goal_rgb_sha256')
        receipt.update(representation='state28',rendering_performed=False,visual_alpha_exception=False,
            reference_reset_file_sha256=row['reset_file_sha256'],replay_source_sha256=REPLAY_SHA,
            legacy_helper_sha256=LEGACY_SHA)
        legacy.write(attempt/'pairing.json',receipt)
    paired=replay.ReplayPairWrapper(env,reference,actor,lambda u,state:goal.copy(),
        snapshot_fn=legacy.physical_snapshot,model_snapshot_fn=legacy.model_snapshot,on_reset=on_reset)
    # Gym's recorder requires a Gym Env. Bind existing methods rather than reimplementing them.
    proxy=gym.Wrapper(env)
    proxy.reset=paired.reset; proxy.step=paired.step; proxy.close=paired.close
    return proxy,actor


def annotate_reset(attempt):
    path=attempt/'reset.json'
    reset=json.loads(path.read_text()); pair=json.loads((attempt/'pairing.json').read_text())
    assert pair['strict_initial_and_goal_match'] and pair['policy_warmup_steps']==0
    reset['pairing_limitation']='explicit sealed reference replay; initial/goal/ctrl/mocap/private/RNG/model checked exactly'
    reset['reference_replay']=pair
    import hashlib
    tmp=path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(reset,allow_nan=False)); tmp.replace(path)
    return hashlib.sha256((attempt/'pairing.json').read_bytes()).hexdigest()
