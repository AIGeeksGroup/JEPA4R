"""Reuse the frozen serial runner, with a new control and log namespace."""
import hashlib
import importlib.util
from pathlib import Path
import time
import traceback

HERE = Path(__file__).resolve().parent
source = HERE/'serial_relay.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'ff7eb51417c8301f27987743701d4a0336117c356904443bb174ab84475366cb'
spec = importlib.util.spec_from_file_location('_existing_cube2_serial_relay', source)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.HERE = HERE/'control_v2'
old.RUN = HERE/'runtime_v2'
code = 1
try:
    old.main()
    code = 0
except BaseException:
    if old.RUN.exists():
        old.save('failure.json',dict(actual_exit_code=1,error=traceback.format_exc(),at=time.time()))
        old.save('status.json',dict(stage='failed',actual_exit_code=1,updated=time.time()))
    raise
finally:
    if old.RUN.exists():
        (old.RUN/'outer_exit_code').write_text(str(code)+'\n')
