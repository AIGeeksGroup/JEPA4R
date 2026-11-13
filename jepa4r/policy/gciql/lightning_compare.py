# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/lightning_compare.py (verbatim except: imports / paths adapted
# to the package; lines wrapped at 120 columns)
"""Full sealed Lightning-state comparison with explicit run-local metadata aliases."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
from jepa4r.policy.gciql.gates.arrays import array_identity,compare_nested

def normalise(state):
    result=dict(state);run=Path(state['spt_run_dir']).resolve();aliases=[]
    def local(value,field):
        if value=='':return value
        if not isinstance(value,str):raise ValueError('checkpoint metadata path must be a string')
        path=Path(value).resolve()
        if not path.is_relative_to(run):raise ValueError('checkpoint path is outside its actual SPT run')
        alias='$SPT_RUN/'+path.relative_to(run).as_posix()
        aliases.append(dict(field=field,actual=value,canonical=alias))
        return alias
    aliases.append(dict(field='spt_run_dir',actual=state['spt_run_dir'],canonical='$SPT_RUN'))
    result['spt_run_dir']='$SPT_RUN';result['callbacks']=copy.deepcopy(state['callbacks'])
    sentinels=[]
    for name,row in result['callbacks'].items():
        if not name.startswith('ModelCheckpoint'):continue
        for field in ('dirpath','best_model_path','kth_best_model_path','last_model_path'):
            if field in row:row[field]=local(row[field],'callbacks/'+name+'/'+field)
        if 'best_k_models'in row:
            original = row['best_k_models']
            mapped = {local(path, 'callbacks/' + name + '/best_k_models'): score for path, score in original.items()}
            if len(mapped)!=len(original):raise ValueError('checkpoint path aliases collide')
            row['best_k_models']=mapped
        if 'kth_value'in row:
            value=np.asarray(row['kth_value'])
            if value.dtype.kind!='f' or value.shape!=():raise ValueError('unknown checkpoint selection sentinel')
            if np.isnan(value):raise ValueError('NaN checkpoint selection metadata is not a valid sentinel')
            if np.isinf(value):
                identity=dict(dtype=str(value.dtype),shape=list(value.shape),sha256=array_identity(value))
                row['kth_value'] = dict(
                    scope='original Lightning checkpoint selection infinity sentinel', identity=identity
                )
                sentinels.append(dict(field='callbacks/'+name+'/kth_value',identity=identity))
    return result,dict(path_aliases=aliases,selection_sentinels=sentinels,no_checkpoint_fields_dropped=True)

def compare(actual_record,reference_record,expected_identity):
    import torch
    from jepa4r.utils.training_state import validate_identity
    from .gate_checks import pinned,_state_cpu_arrays
    def load(record):
        path=pinned(record);meta=json.loads(path.with_suffix('.ckpt.json').read_text())
        if (meta.get('schema')!='sealed-lightning-recovery-v1'or meta.get('phase')!='complete'or
            meta.get('actual_exit_code')!=0 or meta.get('global_step')!=2 or
            hashlib.sha256(path.read_bytes()).hexdigest()!=meta.get('sha256')):
            raise ValueError('full-state comparison requires the sealed final two-update checkpoint')
        validate_identity(expected_identity,meta['identity'])
        state=torch.load(path,map_location='cpu',weights_only=False)
        if state.get('global_step')!=2:raise ValueError('actual Lightning global step differs')
        validate_identity(expected_identity,state['cube_smoke_state']['identity'])
        required = {
            'state_dict',
            'optimizer_states',
            'lr_schedulers',
            'loops',
            'callbacks',
            'cube_smoke_state',
            'adroit_train_loader_state',
            'spt_run_dir',
        }
        if not required<=set(state):raise ValueError('complete Lightning training state is missing a field')
        return normalise(_state_cpu_arrays(state))
    actual,a_aliases=load(actual_record);reference,r_aliases=load(reference_record)
    result=compare_nested(actual,reference,exact=True)
    result.update(normalisations=dict(actual=a_aliases,reference=r_aliases),
        compared_top_level_fields=sorted(reference),expected_identity=expected_identity)
    return result
