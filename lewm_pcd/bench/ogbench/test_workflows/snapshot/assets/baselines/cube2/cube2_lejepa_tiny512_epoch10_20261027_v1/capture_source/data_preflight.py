"""Small sealed-index hash plus 125 RGB source stats; never reads raw HDF5 payloads."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def inspect_sources(index_path, expected_sha):
    raw = index_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('The small sealed training-view index changed')
    sources = {name: item for name, item in json.loads(raw)['raw_data_files'].items()
               if name.endswith('_pix.h5')}
    if len(sources) != 125:
        raise ValueError(f'Expected exactly 125 sealed RGB source files, got {len(sources)}')
    train_sources = {name for name in sources if '_train_part' in Path(name).name}
    validation_sources = {name for name in sources if '_validation_part' in Path(name).name}
    if len(train_sources) != 100 or len(validation_sources) != 25 or train_sources | validation_sources != set(sources):
        raise ValueError('Expected exactly 100 train and 25 validation RGB sources')
    counts = dict(total=125, present=0, missing=0, size_match=0, size_mismatch=0)
    split_counts = {split: dict(total=count, present=0, missing=0, size_match=0, size_mismatch=0)
                    for split, count in [('train', 100), ('validation', 25)]}
    differences = []
    for name, expected in sources.items():
        split = split_counts['train' if name in train_sources else 'validation']
        try:
            actual_bytes = Path(name).stat().st_size
        except FileNotFoundError:
            counts['missing'] += 1
            split['missing'] += 1
            differences.append({'path': name, 'exists': False, 'expected_bytes': expected['bytes']})
            continue
        counts['present'] += 1
        split['present'] += 1
        if actual_bytes == expected['bytes']:
            counts['size_match'] += 1
            split['size_match'] += 1
        else:
            counts['size_mismatch'] += 1
            split['size_mismatch'] += 1
            differences.append({'path': name, 'exists': True, 'expected_bytes': expected['bytes'],
                                'actual_bytes': actual_bytes})
    return {'stage': 'sealed_rgb_source_stat_only',
            'observed_at_utc': datetime.now(timezone.utc).isoformat(),
            'index_path': str(index_path), 'index_sha256': expected_sha,
            'counts': counts, 'split_counts': split_counts, 'differences': differences,
            'metadata_ready': counts['missing'] == 0 and counts['size_mismatch'] == 0,
            'training_metadata_ready': split_counts['train']['size_match'] == 100,
            'validation_metadata_ready': split_counts['validation']['size_match'] == 25,
            'required_for_native_trainer': '100 train sources; 25 unused validation sources are recorded only',
            'payload_or_loader_validation': 'not performed by this metadata-only guard'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    plan = json.loads((Path(__file__).resolve().parent / 'plan.json').read_text())
    result = inspect_sources(Path(plan['data']['sealed_view_index']),
                             plan['data']['sealed_view_index_sha256'])
    rendered = json.dumps(result, indent=2) + '\n'
    if args.report:
        with args.report.open('x') as report:
            report.write(rendered)
    print(rendered, end='')
    return 0 if result['training_metadata_ready'] else 3


if __name__ == '__main__':
    raise SystemExit(main())
