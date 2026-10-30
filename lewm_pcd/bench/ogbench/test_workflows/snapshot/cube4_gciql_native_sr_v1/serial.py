"""Serial eight-stage Quad SR pipeline; preserve the accepted process/resource guard."""
import argparse
import hashlib
from pathlib import Path
import time
import traceback
import types

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
HERE=ROOT/'cube4_gciql_native_sr_v1';RUN=HERE/'runtime_v1'
SOURCE=ROOT/'cube2_official_embeddings_v1/serial_relay.py'
SHA='ff7eb51417c8301f27987743701d4a0336117c356904443bb174ab84475366cb'
EDITS=[("plan['cube_count'] == 2 and plan['data_id'] == 'cube-double-play-v0'","plan['cube_count'] == 4 and plan['data_id'] == 'cube-quadruple-play-v0'"),
 ("cube_count=2, data_id='cube-double-play-v0'","cube_count=4, data_id='cube-quadruple-play-v0'"),
 ('expected_features_per_modality=1101100','expected_features_per_modality=0'),
 ("plan['training_updates'] == plan['SR_episodes'] == 0","plan['training_updates'] == 0 and plan['SR_episodes'] == 500"),
 ('training_updates=0, SR_episodes=0','training_updates=0, SR_episodes=500'),
 ("stage='all_embedding_phases_completed'","stage='Native_SR_pipeline_complete'")]


def load(source=SOURCE):
    raw=Path(source).read_bytes();assert hashlib.sha256(raw).hexdigest()==SHA
    original=raw.decode();effective=original
    for before,after in EDITS:
        assert effective.count(before)==1;effective=effective.replace(before,after)
    restored=effective
    for before,after in reversed(EDITS):restored=restored.replace(after,before)
    assert restored==original
    module=types.ModuleType('_original_Quad_Native_serial');module.__file__=str(source)
    exec(compile(effective,str(source),'exec'),module.__dict__);module.HERE=HERE;module.RUN=RUN
    return module


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:print('prepared_only; no Native evaluation started')
    else:
        old=load();owned=not RUN.exists();code=1
        try:old.main();code=0
        except BaseException:
            if owned and RUN.exists():
                old.save('failure.json',dict(actual_exit_code=1,error=traceback.format_exc(),at=time.time()))
                old.save('status.json',dict(stage='failed',actual_exit_code=1,updated=time.time()))
            raise
        finally:
            if owned and RUN.exists():
                with (RUN/'outer_exit_code').open('x') as f:f.write(str(code)+'\n')
