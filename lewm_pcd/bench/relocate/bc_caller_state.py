"""SHA-bound caller bookkeeping beside the original-loop BC checkpoints."""
import hashlib
import json
import os
from pathlib import Path


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):digest.update(block)
    return digest.hexdigest()


def _main(checkpoint,updates):
    path=Path(checkpoint);meta=path.with_suffix(path.suffix+'.json')
    row=json.loads(meta.read_text())
    if (row.get('actual_exit_code')!=0 or row['sha256']!=_sha(path)
            or row['progress']['committed_updates']!=updates):
        raise ValueError('BC caller/main checkpoint progress or SHA differs')
    return path,meta


def save_caller_state(checkpoint,*,identity,state,committed_updates,model_sha256):
    path,meta=_main(checkpoint,committed_updates)
    row=dict(schema='jepa4r-bc-caller-state-v1',checkpoint_sha256=_sha(path),
        checkpoint_metadata_sha256=_sha(meta),identity=identity,committed_updates=committed_updates,
        model_sha256=model_sha256,state=state)
    sidecar=path.with_suffix(path.suffix+'.caller.json')
    with sidecar.open('x') as stream:
        json.dump(row,stream,indent=2,allow_nan=False);stream.write(chr(10))
        stream.flush();os.fsync(stream.fileno())
    return sidecar


def load_caller_state(checkpoint,*,identity,committed_updates,model_sha256,adam_steps):
    path,meta=_main(checkpoint,committed_updates)
    row=json.loads(path.with_suffix(path.suffix+'.caller.json').read_text())
    if (row.get('schema')!='jepa4r-bc-caller-state-v1' or row['identity']!=identity
            or row['checkpoint_sha256']!=_sha(path) or row['checkpoint_metadata_sha256']!=_sha(meta)
            or row['committed_updates']!=committed_updates or row['model_sha256']!=model_sha256
            or not adam_steps or any(step!=committed_updates for step in adam_steps)):
        raise ValueError('BC caller history does not match restored source/model/Adam/progress')
    return row['state']


def epoch_metric_prefix(records):
    rows=[]
    for index,event in enumerate(records):
        if event['epoch']!=index or event['minibatches']<1:raise ValueError('BC restored epoch prefix differs')
        rows.append(dict(epoch_loss=event['running_loss']/event['minibatches'],_step=index+1))
        if 'evaluation' in event:rows.append(dict(event['evaluation'],_step=index+1))
    return rows
