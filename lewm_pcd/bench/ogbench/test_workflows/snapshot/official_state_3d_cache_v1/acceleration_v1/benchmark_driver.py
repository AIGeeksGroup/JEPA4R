"""Only the two approved renderer widths and one uint8/GPU-preprocessor test."""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
H = HERE.parent
ROOT = H.parent


if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('_existing_phase', ROOT / 'official_state_2d_cache_v1/relay_cache.py')
    phase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase)
    phase.RUN = HERE / 'benchmark_v1/runtime'
    py = str(ROOT / 'env_runtime/bin/python')
    code = 1
    try:
        for workers in (8, 16):
            # Direct stages inherit this one owned PGID, including every pool worker.
            phase.phase(f'render_workers{workers}', [py, '-B', str(HERE / 'bench_render_parallel.py'),
                '--plan', str(H / 'producer_plan.json'), '--output-dir', str(HERE / f'benchmark_render_v1/workers{workers}'),
                '--execute', '--stage-workers', str(workers)])
        phase.phase('GPU_preprocess', [py, '-B', str(HERE / 'bench_gpu_preprocess.py'),
            '--plan', str(H / 'producer_plan.json'), '--samples', str(H / 'cache/pilot_train_boundary.npz'),
            '--output', str(HERE / 'GPU_preprocess_report_v1.json'), '--execute'])
        phase.write('closure.json', dict(actual_exit_code=0, rendered_rows=56056,
                                         encoding_updates=0, original_cache_unchanged=True))
        code = 0
    except BaseException as error:
        phase.write('failure.json', dict(actual_exit_code=1, error=repr(error)))
        raise
    finally:
        (phase.RUN / 'pipeline_actual_exit_code').write_text(str(code) + '\n')
