"""Independent read-only audit of frozen, already executed comparison artifacts."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'review_evidence_actual_v1'
LABELS = ('baseline_A', 'prefetch1', 'workers2', 'workers4', 'baseline_B')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def require_exit(label):
    value = read(label + '_exit.json')
    assert value['actual_exit_code'] == 0 and value['residual_members'] == [], label
    return value


def batch_summary(batch):
    return {key: dict(shape=list(value.shape), dtype=value.dtype.str, bytes=value.nbytes,
                     sha256=hashlib.sha256(value.tobytes()).hexdigest())
            for key, value in batch.items()}


def numerical():
    exits = {key: require_exit(key) for key in ('correctness', 'fresh')}
    c, f = read('correctness_report.json'), read('fresh_report.json')
    a, freeze = read('ACCEPTANCE.json'), read('stat-freeze.json')
    source = read('run_source_freeze_v1.json')
    assert a['stage'] == 'complete' and a['actual_exit_code'] == freeze['actual_exit_code'] == 0
    assert sha(EVIDENCE / 'ACCEPTANCE.json') == source['data_acceptance_sha256'] == freeze['acceptance_sha256']
    assert sha(EVIDENCE / 'stat-freeze.json') == source['data_stat_freeze_sha256']
    for name, record in source['files'].items():
        assert sha(HERE / name) == record['sha256'] and (HERE / name).stat().st_size == record['bytes'], name
    for report in (c, f):
        assert report['actual_exit_code'] == 0 and not report['fixture_only']
        assert not report['artificial_terminal'] and not report['formal_training'] and not report['SR']
        assert report['harness_sha256'] == source['files']['measure.py']['sha256']
        assert report['adapter_sha256'] == source['files']['ordered_input.py']['sha256']
        assert report['data_receipt']['sha256'] == source['data_acceptance_sha256']
        assert report['data_stat_freeze']['sha256'] == source['data_stat_freeze_sha256']
        assert (report['current_dim'], report['goal_dim'], report['action_dim'], report['batch_size']) == (46, 46, 5, 1024)
        assert report['input_dtype'] == 'float32' and report['official_config']['alpha'] == 1.0
        assert report['raw_npz_stats_before'] == report['raw_npz_stats_after']
    assert c['source_sha256'] == f['source_sha256'] and c['official_config'] == f['official_config']
    for split, name, episodes, rows, valids in (
            ('train', 'cube-triple-play-v0.npz', 3000, 3003000, 3000000),
            ('validation', 'cube-triple-play-v0-val.npz', 300, 300300, 300000)):
        data, frozen = c['datasets'][name], freeze['datasets'][split]
        accepted = next(x for x in a['datasets'] if x['schema_loader']['path'] == frozen['path'])
        assert (data['episodes'], data['rows'], data['compact_valid_rows']) == (episodes, rows, valids)
        assert data['raw_sha256'] == accepted['download']['sha256'] == frozen['raw_sha256']
        assert data['accepted_stat'] == frozen['accepted_stat']
        assert all(c['raw_npz_stats_before'][name][key] == value for key, value in frozen['accepted_stat'].items())
        assert f['datasets'][name] == data
    assert c['actual_updates_per_mode'] == 3 and c['unique_checkpoint_step'] == 3
    assert list(c['modes']) == ['baseline', 'prefetch1', 'workers2', 'workers4']
    baseline = c['modes']['baseline']
    for mode, result in c['modes'].items():
        assert len(result['updates']) == 3 and result['Adam_count'] == 3 and result['network_step'] == 4
        for key in ('all_batch_keys_dtype_shape_bytes_exact', 'raw_loss_gradients_exact',
                    'update_info_exact', 'boundary_NumPy_RNG_exact', 'final_full153_exact'):
            assert result[key] is True, (mode, key)
        for key in ('updates', 'validation', 'consumed_training_rng', 'validation_rng', 'final_rng', 'final_state'):
            assert result[key] == baseline[key], (mode, key)
        assert result['final_state'] == c['saved_state'] and result['final_state']['leaves'] == 153
        assert result['threads_after_close']['row_count'] == result['threads_after_close']['coordinator_count'] == 0
        assert all(math.isfinite(v) for update in result['updates'] for v in update.values())
    assert f['reference_report_sha256'] == sha(EVIDENCE / 'correctness_report.json')
    assert f['full153_restored_exact'] == c['saved_state']
    assert f['Adam_count'] == 3 and f['network_step'] == 4 and f['finite_actions']
    assert f['restored_agent_JAX_RNG'] and f['host_NumPy_RNG_restored_from_separate_aux']
    assert not f['original_checkpoint_contains_host_NumPy_RNG'] and f['next_training_batch_digest_exact']
    with np.load(EVIDENCE / 'reload_batch.npz', allow_pickle=False) as saved:
        actual = batch_summary(dict(saved))
    assert actual == c['golden_batches']['train3']
    with np.load(EVIDENCE / 'numpy_tail_aux.npz', allow_pickle=False) as z:
        tail = dict(generator=str(z['generator'].item()), keys_sha256=hashlib.sha256(z['keys'].tobytes()).hexdigest(),
                    position=int(z['position']), has_gauss=int(z['has_gauss']), cached_gaussian=float(z['cached_gaussian']))
    assert tail == c['final_NumPy_RNG'] == baseline['final_rng']
    for label in ('correctness', 'fresh'):
        argv = read(label + '_argv.json')['argv']
        assert '-O' not in argv and argv[argv.index('--stage') + 1] == label
        assert argv[argv.index('--data-stat-freeze') + 1] == c['data_stat_freeze']['path']
    return dict(status='verified', parent_child_exits=exits, modes=list(c['modes']),
        train_episodes=3000, validation_episodes=300, updates_per_mode=3,
        loss_info_max_abs_difference=0.0, final_state=c['saved_state'],
        saved_reload_batch_independently_rehashed=True, NumPy_aux_independently_rehashed=True,
        tensor_assertions='Executed frozen harness compares actual batches, raw gradients and all153 leaves; child exits0',
        checkpoint_step=3, fresh_agent_and_host_rng_next_batch_verified=True,
        official_source_sha256=c['source_sha256'], data_acceptance_sha256=sha(EVIDENCE / 'ACCEPTANCE.json'),
        stat_freeze_sha256=sha(EVIDENCE / 'stat-freeze.json'), device=c['devices'])


def performance():
    gpu = [json.loads(line) for line in (EVIDENCE / 'GPU.jsonl').read_text().splitlines() if line]
    resources = [json.loads(line) for line in (EVIDENCE / 'prepare_resources.jsonl').read_text().splitlines() if line]
    reports, results = {}, {}
    base = None
    for label in LABELS:
        require_exit(label)
        report = read(label + '_report.json')
        reports[label] = report
        assert report['actual_exit_code'] == 0 and report['total_updates'] == 6208
        assert report['stage'] == 'benchmark'
        assert report['mode'] == ('baseline' if label.startswith('baseline_') else label)
        assert report['warmup_updates'] == 64 and report['timed_updates'] == 6144
        assert report['Adam_count'] == 6208 and report['network_step'] == 6209
        assert report['full_agent_and_info_tail_synchronized'] and report['sample_wait_and_dispatch_included']
        assert report['raw_npz_stats_before'] == report['raw_npz_stats_after']
        if base is None:
            base = report
        for key in ('source_sha256', 'harness_sha256', 'adapter_sha256', 'official_config', 'data_receipt',
                    'data_stat_freeze', 'datasets', 'final_NumPy_RNG', 'CPU_affinity',
                    'environment', 'devices', 'JAX', 'NumPy', 'Flax', 'JAX_default_matmul_precision'):
            assert report[key] == base[key], (label, key)
        hz, utilization, memory, windows = [], [], [], []
        assert len(report['windows']) == 3
        for index, w in enumerate(report['windows']):
            assert w['window'] == index + 1 and w['updates'] == 2048 and w['seconds'] > 0
            rate = w['updates'] / w['seconds']
            assert math.isclose(rate, w['Hz'], rel_tol=1e-13)
            assert w['last_batch'] == base['windows'][index]['last_batch']
            assert w['boundary_NumPy_RNG'] == base['windows'][index]['boundary_NumPy_RNG']
            samples = [x for x in gpu if w['wall_started'] <= x['at'] <= w['wall_finished'] and x['actual_exit_code'] == 0]
            parsed = [list(map(float, x['GPU_csv'].split(','))) for x in samples]
            assert parsed and all(len(x) == 4 for x in parsed)
            utilization.extend(x[0] for x in parsed)
            memory.extend(x[1] for x in parsed)
            hz.append(rate)
            windows.append(dict(Hz=rate, seconds=w['seconds'], GPU_samples=len(parsed),
                                GPU_utilization_mean=statistics.mean(x[0] for x in parsed),
                                host_sample_seconds=w['host_sample_seconds']))
        own = [x for x in resources if x['phase'] == label]
        assert own
        event_ranges = {k: max(x['events'][k] for x in own) - min(x['events'][k] for x in own)
                        for k in ('max', 'oom', 'oom_kill')}
        assert all(value == 0 for value in event_ranges.values())
        results[label] = dict(median_Hz=statistics.median(hz), windows=windows,
            GPU_utilization_mean=statistics.mean(utilization), GPU_utilization_median=statistics.median(utilization),
            GPU_memory_peak_MiB=max(memory), own_RSS_peak_GiB=max(x['RSS_sum'] for x in own) / 1024**3,
            memory_event_deltas=event_ranges, final153_fingerprint=report['final_state'],
            window_info_max_abs_difference_vs_A=[max(abs(value - base['windows'][i]['last_metrics'][key])
                for key, value in window['last_metrics'].items()) for i, window in enumerate(report['windows'])])
    a, b = (results[x]['median_Hz'] for x in ('baseline_A', 'baseline_B'))
    drift = abs(b / a - 1)
    for label in LABELS[1:-1]:
        r = results[label]
        r['gain_vs_A_percent'] = 100 * (r['median_Hz'] / a - 1)
        r['gain_vs_B_percent'] = 100 * (r['median_Hz'] / b - 1)
        r['meets_5pct_against_both_controls'] = r['median_Hz'] >= 1.05 * max(a, b)
    winners = [label for label in LABELS[1:-1] if results[label]['meets_5pct_against_both_controls']]
    return dict(status='verified', results=results, baseline_control_drift_percent=100 * drift,
        control_within_5pct=drift <= .05, prospective_winners=winners,
        selection='retain baseline' if not winners or drift > .05 else 'literal baseline confirmation required before adoption',
        baseline_scope='Same thin-interface baseline adds caller check and NumPy state snapshot to official sample',
        final_NumPy_rng_identical_all_five=True,
        final153_identical_all_five=len({r['final_state']['sha256'] for r in reports.values()}) == 1,
        model_equivalence_limit='Same-process3-update strict gate passes; independent6208-update states differ including A/B controls. Cause unverified; no6208 model equivalence claim.',
        timing_scope='End-to-end bounded diagnostic; formal validation/evaluation/logging excluded; no1M ETA or SR')


def literal_control(performance_result):
    require_exit('literal')
    assert (EVIDENCE / 'literal_actual_exit_code').read_text().strip() == '0'
    literal, base = read('literal_report.json'), read('baseline_A_report.json')
    assert literal['actual_exit_code'] == 0 and literal['literal_official_sampler_control'] is True
    assert literal['control_wrapper_sha256'] == sha(HERE / 'literal_baseline.py')
    for key in ('stage', 'mode', 'warmup_updates', 'timed_updates', 'total_updates', 'Adam_count',
                'network_step', 'source_sha256', 'harness_sha256', 'adapter_sha256', 'official_config',
                'data_receipt', 'data_stat_freeze', 'datasets', 'CPU_affinity', 'environment', 'devices',
                'JAX', 'NumPy', 'Flax', 'JAX_default_matmul_precision', 'final_NumPy_RNG'):
        assert literal[key] == base[key], key
    assert literal['raw_npz_stats_before'] == literal['raw_npz_stats_after'] == base['raw_npz_stats_before']
    rates = []
    for i, window in enumerate(literal['windows']):
        assert window['updates'] == 2048 and window['last_batch'] == base['windows'][i]['last_batch']
        assert window['boundary_NumPy_RNG'] == base['windows'][i]['boundary_NumPy_RNG']
        rates.append(window['updates'] / window['seconds'])
    median = statistics.median(rates)
    assert math.isclose(median, literal['median_Hz'], rel_tol=1e-13)
    resources = [json.loads(line) for line in (EVIDENCE / 'literal_resources.jsonl').read_text().splitlines() if line]
    event_ranges = {key: max(x['events'][key] for x in resources) - min(x['events'][key] for x in resources)
                    for key in ('max', 'oom', 'oom_kill')}
    assert all(value == 0 for value in event_ranges.values())
    result = performance_result['results']
    fastest_control = max(median, result['baseline_A']['median_Hz'], result['baseline_B']['median_Hz'])
    for label in LABELS[1:-1]:
        result[label]['gain_vs_literal_percent'] = 100 * (result[label]['median_Hz'] / median - 1)
        result[label]['gain_vs_fastest_control_percent'] = 100 * (result[label]['median_Hz'] / fastest_control - 1)
        result[label]['meets_5pct_against_all_controls'] = result[label]['median_Hz'] >= 1.05 * fastest_control
    candidates = [label for label in LABELS[1:-1] if result[label]['meets_5pct_against_all_controls']]
    selected = max(candidates, key=lambda label: result[label]['median_Hz']) if candidates else 'baseline'
    performance_result['literal_control'] = dict(median_Hz=median, windows_Hz=rates,
        wrapper_sha256=literal['control_wrapper_sha256'], own_RSS_peak_GiB=max(x['RSS_sum'] for x in resources) / 1024**3,
        memory_event_deltas=event_ranges, GPU_utilization='not sampled for this single control',
        all_window_batch_and_NumPy_rng_digests_match_original=True, final153_fingerprint=literal['final_state'])
    performance_result['selection'] = selected
    performance_result['selection_scope'] = 'Performance selection under declared3-update gate; official default remains unchanged; no formal run started'
    performance_result['fastest_control_Hz'] = fastest_control
    return performance_result


def markdown(result):
    performance_result = result['performance']
    lines = ['# Independent actual-artifact review', '',
        'Verified: complete official train/validation data (3000/300 episodes), four-mode same-process three-update numerical gate, real validation, checkpoint3 and fresh agent/NumPy next-batch restore. All actual exits are0, with no residual child processes.', '',
        '| Input mode | Median updates/s | Timed GPU utilization mean | Peak own RSS GiB |',
        '|---|---:|---:|---:|']
    for label, value in performance_result['results'].items():
        lines.append(f"| {label} | {value['median_Hz']:.3f} | {value['GPU_utilization_mean']:.2f}% | {value['own_RSS_peak_GiB']:.3f} |")
    literal = performance_result['literal_control']
    lines += [f"| literal original sampler | {literal['median_Hz']:.3f} | not sampled | {literal['own_RSS_peak_GiB']:.3f} |", '',
        f"Selected candidate: {performance_result['selection']}. Its median throughput exceeds the fastest control by {performance_result['results'][performance_result['selection']]['gain_vs_fastest_control_percent']:.3f}%. Original control A/B drift is {performance_result['baseline_control_drift_percent']:.3f}%.", '',
        'The two original controls use the same thin input interface, including a caller check and NumPy state snapshot per batch. The literal control calls the original Dataset.sample directly. Every benchmark uses64 warmup plus3x2048 timed updates and synchronizes the whole agent and info at each window tail.', '',
        'Verified: every window-tail batch/RNG digest and final host NumPy RNG agree across all six timing runs. No max/oom/oom_kill increments were observed. GPU means use only monitor samples inside actual timing windows.', '',
        'Not proven:6208-update cross-process model equivalence. All independent final153 fingerprints differ, including original A/B controls. The cause is unverified. Same-process three-update raw loss/gradient/info/final153 comparisons remain exactly equal; no tolerance was relaxed.', '',
        'This selects a performance candidate under the declared short gate. Official defaults remain unchanged. No formal1M training or SR was started.']
    return '\n'.join(lines) + '\n'


def main():
    result = dict(audited_at=time.time(), numerical=numerical(), formal_training=False, SR=False)
    if all((EVIDENCE / (label + '_report.json')).exists() and (EVIDENCE / (label + '_exit.json')).exists()
           for label in LABELS) and (EVIDENCE / 'GPU.jsonl').exists():
        assert (EVIDENCE / 'experiment_actual_exit_code').read_text().strip() == '0'
        result['performance'] = performance()
        if (EVIDENCE / 'literal_report.json').exists() and (EVIDENCE / 'literal_exit.json').exists():
            result['performance'] = literal_control(result['performance'])
            name = 'review_actual_v1.json'
        else:
            name = ('review_actual_v1_controls.json' if result['performance']['prospective_winners']
                    else 'review_actual_v1.json')
    else:
        result['performance'] = dict(status='pending_all_five_actual_exits')
        name = 'review_actual_v1_numerical.json'
    result['evidence_sha256'] = {path.name: sha(path) for path in EVIDENCE.iterdir() if path.is_file()}
    result['reviewer_script_sha256'] = sha(Path(__file__))
    with (HERE / name).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    if name == 'review_actual_v1.json' and 'literal_control' in result['performance']:
        with (HERE / 'review_actual_v1.md').open('x', encoding='utf-8') as stream:
            stream.write(markdown(result))
    print(json.dumps({key: value for key, value in result.items() if key != 'evidence_sha256'}, indent=2))


if __name__ == '__main__':
    main()
