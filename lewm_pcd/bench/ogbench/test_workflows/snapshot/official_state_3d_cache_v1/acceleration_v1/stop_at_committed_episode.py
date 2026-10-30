"""Stop only this cache PGID after a committed status update; retain all outputs."""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import struct
import time

HERE = Path(__file__).resolve().parent
H = HERE.parent
ROOT = H.parent


def cmdline(pid):
    return [v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]


if __name__ == '__main__':
    out = HERE / 'controlled_stop_v1'
    out.mkdir(exist_ok=False)
    stage = json.loads((H / 'full_v1/runtime/full_argv.json').read_text())
    outer = json.loads((H / 'full_v1/runtime/actual_argv.json').read_text())
    pid, pgid = stage['pid'], outer['owned_pgid']
    assert cmdline(pid) == stage['argv']
    assert cmdline(pgid) == outer['argv'] and os.getpgid(pid) == pgid
    assert stage['argv'][2] == str(H / 'produce.py') and stage['argv'][-1] == '--execute'
    assert hashlib.sha256((H / 'produce.py').read_bytes()).hexdigest() == '7e4c201bb259a005e6b3cc49e4a2df3b3963e347c4a6aa26b17ea22666dce4db'
    spec = importlib.util.spec_from_file_location('_owned_process_members', ROOT / 'prepare_runtime.py')
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    baseline = json.loads((H / 'cache/status.json').read_text())
    assert baseline['stage'] == 'encoding' and baseline['failed_episodes'] == 0
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    assert fd >= 0
    assert libc.inotify_add_watch(fd, os.fsencode(H / 'cache'), 0x80) >= 0
    events, attempts, paused = [], [], False
    deadline = time.monotonic() + 55
    try:
        while time.monotonic() < deadline:
            if not select.select([fd], [], [], min(10, deadline - time.monotonic()))[0]:
                continue
            payload = os.read(fd, 65536)
            offset, matched = 0, False
            while offset < len(payload):
                watch, mask, cookie, length = struct.unpack_from('iIII', payload, offset)
                name = payload[offset + 16:offset + 16 + length].split(b'\0')[0]
                offset += 16 + length
                matched |= name == b'status.json'
            if not matched:
                continue
            status = json.loads((H / 'cache/status.json').read_text())
            if status['stage'] != 'encoding' or status['completed_episodes'] <= baseline['completed_episodes']:
                continue
            assert os.getpgid(pid) == pgid and cmdline(pid) == stage['argv']
            os.kill(pid, signal.SIGSTOP)
            paused = True
            for _ in range(100):
                state = next(s for s in Path(f'/proc/{pid}/status').read_text().splitlines() if s.startswith('State:'))
                if 'T' in state:
                    break
                time.sleep(.01)
            assert 'T' in state
            bad = []
            receipts = []
            for split in ('train', 'val'):
                for directory in (H / 'cache' / split).glob('episode_*'):
                    bad.extend(str(p) for p in directory.glob('latent.partial_*.npy'))
                    if (directory / 'latent.npy').exists() and not (directory / 'receipt.json').exists():
                        bad.append(str(directory / 'latent.npy'))
                    if (directory / 'receipt.json').exists():
                        raw = (directory / 'receipt.json').read_bytes()
                        receipts.append(dict(path=str(directory / 'receipt.json'), sha256=hashlib.sha256(raw).hexdigest()))
            attempts.append(dict(status=status, uncommitted_paths=bad, at=time.time()))
            if bad:
                os.kill(pid, signal.SIGCONT)
                paused = False
                assert len(attempts) < 3, 'No clean commit boundary; left original producer running'
                continue
            events.append(dict(signal='SIGSTOP', pid=pid, at=time.time(), purpose='verified clean committed EP boundary'))
            break
        assert paused, 'No committed boundary observed; no termination authorized'
        before = dict(reason='user requested acceleration switch', baseline_status=baseline,
                      clean_status=attempts[-1]['status'], preserved_receipts=receipts,
                      producer_actual_argv=stage, owned_phase_argv=outer, attempts=attempts)
        (out / 'before_stop.json').write_text(json.dumps(before, indent=2) + '\n')
        os.killpg(pgid, signal.SIGINT)
        events.append(dict(signal='SIGINT', pgid=pgid, at=time.time()))
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        paused = False
        for limit, next_signal in ((20, signal.SIGTERM), (5, signal.SIGKILL), (3, None)):
            until = time.monotonic() + limit
            while time.monotonic() < until and guard.members(pgid):
                time.sleep(.2)
            if not guard.members(pgid):
                break
            if next_signal is not None:
                os.killpg(pgid, next_signal)
                events.append(dict(signal=next_signal.name, pgid=pgid, at=time.time()))
        residual = guard.members(pgid)
        assert not residual, residual
        for item in receipts:
            assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
        result = dict(actual_stop_helper_exit_code=0, reason=before['reason'], events=events,
                      preserved_episode_receipts=len(receipts), every_existing_receipt_SHA_unchanged=True,
                      uncommitted_paths=[], owned_residual_members=residual, completed=time.time(),
                      old_training_or_algorithm_failure=False, old_exits_not_rewritten=True)
        (out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    finally:
        if paused:
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
        os.close(fd)
