"""A fresh attempt namespace for the unchanged frozen closed-SR selector."""
import hashlib
from pathlib import Path
import types

HERE=Path(__file__).resolve().parent
BASE_SHA='2f0f0ab1436fb9dd4c850c21f26d590a4ae9a338e277077e983932e2508e0e6d'
EDITS=[("'closed_SR_backup_v1/completed_v1'","'closed_SR_backup_v1/completed_v2'"),
       ("'closed_SR_backup_v1/execute_v1'","'closed_SR_backup_v1/execute_v2'")]


def load():
    raw=(HERE/'build_backup.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==BASE_SHA
    original=raw.decode();effective=original
    for before,after in EDITS:
        assert effective.count(before)==1;effective=effective.replace(before,after)
    restored=effective
    for before,after in reversed(EDITS):
        assert restored.count(after)==1;restored=restored.replace(after,before)
    assert restored==original
    module=types.ModuleType('_Quad_closed_SR_fresh_namespace');module.__file__=str(Path(__file__).resolve())
    exec(compile(effective,module.__file__,'exec'),module.__dict__)
    original_collect=module.collect
    def collect(io):
        paths,closure=original_collect(io)
        paths.update(HERE/name for name in ('build_backup_v2.py','copy_backup_v2.py','SOURCE_FREEZE_v2.json','HANDOFF_v2.json'))
        closure['backup_namespace_overlay']=dict(base_sha256=BASE_SHA,strict_substitution_count=2,
            inverse_entire_selector_exact=True,old_failed_execute_and_output_preserved=True)
        return paths,closure
    module.collect=collect
    return module


if __name__=='__main__':raise SystemExit(load().main())
