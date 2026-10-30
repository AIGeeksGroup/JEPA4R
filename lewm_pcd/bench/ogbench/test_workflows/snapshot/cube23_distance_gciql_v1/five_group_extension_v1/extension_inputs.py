"""Cube4 guard and no-lookahead visual interface around frozen input code."""
import hashlib
from pathlib import Path
import sys
import types

from distance_inputs import DistanceInputs, create_agent

STATE_SOURCE=Path(__file__).resolve().parent.parent/'state_priority_v1/state_inputs.py'
STATE_SHA='ada2bdc73de50753a83ee951cb99849c9bd3282f3e6527c4e8d906f85f7d9fd9'
source=STATE_SOURCE.read_bytes()
assert hashlib.sha256(source).hexdigest()==STATE_SHA
before='cubes not in (1,2,3)'
after='cubes not in (1,2,3,4)'
text=source.decode();assert text.count(before)==1
effective=text.replace(before,after)
assert effective.replace(after,before)==text
name='_frozen_state_inputs_cube4_guard_only'
module=types.ModuleType(name);module.__file__=str(STATE_SOURCE);sys.modules[name]=module
exec(compile(effective,str(STATE_SOURCE),'exec'),module.__dict__)
StateInputs=module.StateInputs
STATE_EXTENSION_PROOF=dict(original_source_sha256=STATE_SHA,
    effective_source_sha256=hashlib.sha256(effective.encode()).hexdigest(),
    exact_replacement=[before,after],replacement_count=1,original_source_unmodified=True)


class VisualInputs(DistanceInputs):
    """Existing same-batch four-role lookup has no pending future sample."""
    pending=False

    def sample(self,batch_size,idxs=None,evaluation=False,*,prefetch_next=True):
        return super().sample(batch_size,idxs=idxs,evaluation=evaluation)

    def barrier(self):
        pass
