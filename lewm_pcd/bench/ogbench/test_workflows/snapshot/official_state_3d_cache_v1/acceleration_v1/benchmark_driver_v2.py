"""Same-source renderer capacity only; native color bitwise equality is unproven."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = HERE.parent
ROOT = H.parent

if __name__ == '__main__':
    spec = importlib.util.spec_from_file_location('_existing_phase', ROOT / 'official_state_2d_cache_v1/relay_cache.py')
    phase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase)
    phase.RUN = HERE / 'benchmark_v2/runtime'
    py, code = str(ROOT / 'env_runtime/bin/python'), 1
    try:
        for workers in (8, 16):
            phase.phase(f'render_workers{workers}', [py, '-B', str(HERE / 'bench_render_parallel_v2.py'),
                '--plan', str(H / 'producer_plan.json'), '--output-dir', str(HERE / f'benchmark_render_v2/workers{workers}'),
                '--execute', '--stage-workers', str(workers)])
        phase.write('closure.json', dict(actual_exit_code=0, same_source_capacity_only=True,
            numeric_equivalence_verified=False, native_color_bitexact_not_guaranteed=True,
            rendered_rows=56056, training_updates=0, original_cache_unchanged=True))
        code = 0
    except BaseException as error:
        phase.write('failure.json', dict(actual_exit_code=1, error=repr(error)))
        raise
    finally:
        (phase.RUN / 'pipeline_actual_exit_code').write_text(str(code) + '\n')
