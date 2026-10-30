"""Five new true-width roles reusing the frozen async100 training/save loop."""
import argparse
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
CASE=HERE.parent
PRESERVED={
    'trainer_runtime.py':'18acaead938e573f80ec6e87e0cb44822a734decf2e97730861886841a22b82c',
    'distance_inputs.py':'b9906dc2f4e7a38826f5ee05031713618a6ce408ee5555302801505452758f90',
    'gciql_boundary.py':'a81363583a4826194b90974be2ad62c5895ab6b963e15ba042c9f893a46ba5e6',
    'state_priority_v1/state_runtime.py':'623fff33b0c836c6268257689216cf78de8694fbab1107dddbd4bdbfc3ee1956',
}
for name,expected in PRESERVED.items():
    assert hashlib.sha256((CASE/name).read_bytes()).hexdigest()==expected,name
sys.path[:0]=[str(HERE),str(CASE/'state_priority_v1'),str(CASE)]
import state_runtime as preserved
from extension_inputs import StateInputs, VisualInputs, STATE_EXTENSION_PROOF

GROUPS={'G10':(4,'state',55,12),'G05':(2,'rgb',211,198),
        'G06':(2,'pcd',211,198),'G08':(3,'rgb',211,201),'G09':(3,'pcd',211,201)}


def install(input_mode='baseline',host_metrics='async100'):
    runtime=preserved.install(input_mode,host_metrics)

    def make_input(dataset,cfg,*,cubes,mode,features,workers):
        if mode=='state':
            assert cubes==4 and features is None and workers==0
            selected=input_mode if not getattr(make_input,'made_train',False) else 'baseline'
            make_input.made_train=True
            return StateInputs(dataset,cfg,cubes=cubes,mode=selected,
                ordered_input_path=runtime.ROOT/'official_state_workers_v1/ordered_input.py')
        assert input_mode=='baseline' and cubes in (2,3) and mode in ('rgb','pcd') and workers==4
        return VisualInputs(dataset,cfg,cubes=cubes,mode=mode,features=features,workers=workers)

    def prepare(config,shared_datasets=None):
        expected=GROUPS[config['group_id']]
        assert (config['cube_count'],config['modality'],config['input_contract']['current_dim'],
                config['input_contract']['goal_dim'])==expected
        assert config['save_step1'] is True
        make_input.made_train=False
        samplers,resources,binding=preserved.ORIGINAL_PREPARE(config,shared_datasets)
        resources.actual_update_count=resources.train_batches=0
        def barrier():
            for sampler in samplers.values():sampler.barrier()
            assert not any(sampler.pending for sampler in samplers.values())
        resources.barrier=barrier
        binding['runtime'].update(input_mode=input_mode if config['modality']=='state' else 'samebatch_roles4',
            host_metrics=host_metrics,formal_metric_finite_scope='every100 recorded updates and each5000 validation',
            no_every_update_finite_trace_claim=True,ordered_prefetch_barrier_before_checkpoints=True,
            visual_future_sampling=False,extension_group_id=config['group_id'])
        if config['modality']=='state':binding['runtime']['cube4_guard_extension']=STATE_EXTENSION_PROOF
        paths=[Path(__file__),HERE/'extension_inputs.py',CASE/'state_priority_v1/state_runtime.py',
               CASE/'state_priority_v1/state_inputs.py',runtime.ROOT/'official_state_workers_v1/ordered_input.py']
        binding['code'].update({str(path):runtime.utility.sha(path) for path in paths})
        return samplers,resources,binding

    runtime.DistanceInputs=make_input
    runtime.prepare=prepare
    return runtime


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--input-mode',choices=('baseline','prefetch1'),default='baseline')
    args=parser.parse_args()
    raise SystemExit(install(args.input_mode).run(args.config))
