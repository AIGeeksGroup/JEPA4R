"""Pin and reuse the original M5 thread/physics runner with Quad registration."""
import hashlib
from pathlib import Path

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
SOURCE=ROOT/'cube23_distance_gciql_v1/native_sr_visual_v1/m5_restore_v1/run_group.py'
SOURCE_SHA='2af9fc40e19478ddb91519c74f288bac4389ee6f588248b674a8fa6b317db5d6'
EDITS=[("import visual_sr as core\nfrom cube3_successor_v1 import visual_sr_cube3\nvisual_sr_cube3.install(core)",
        "from quad_bridge import load_core\ncore = load_core()"),
       ("choices=('G05', 'G06', 'G08', 'G09')", "choices=('G11', 'G12')")]


def effective_source(source=SOURCE):
    raw=Path(source).read_bytes();assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA
    original=raw.decode();effective=original
    for before,after in EDITS:
        assert effective.count(before)==1;effective=effective.replace(before,after)
    restored=effective
    for before,after in reversed(EDITS):restored=restored.replace(after,before)
    assert restored==original
    return effective


if __name__=='__main__':
    exec(compile(effective_source(),str(SOURCE),'exec'),dict(__name__='__main__',__file__=str(SOURCE)))
