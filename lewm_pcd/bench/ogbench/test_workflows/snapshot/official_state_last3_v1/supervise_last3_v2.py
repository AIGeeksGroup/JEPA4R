"""Isolated retry; reuse the same frozen evaluation-only owner guard."""
import hashlib
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'supervise_last3.py'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == (
    '710177beb7416aae2736f9d9f8dd64d24a7fa7bd4c6aa49faa5271f8bf4f4842')
spec = importlib.util.spec_from_file_location('_last3_owner_guard_v1', SOURCE)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
ATTEMPT = HERE / 'attempt_v2'
guard.supervisor.HERE = ATTEMPT

if __name__ == '__main__':
    code = 1
    try:
        guard.supervisor.main()
        code = 0
    except BaseException as exc:
        if guard.supervisor.guard.ROOT != HERE.parent:
            guard.write('supervisor_failure.json', dict(error=repr(exc), actual_exit_code=1))
        raise
    finally:
        (ATTEMPT / 'supervisor_actual_exit_code').write_text(str(code) + '\n')
