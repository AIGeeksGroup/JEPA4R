# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/gates/records.py (verbatim)
"""Immutable array receipts; failed producer records remain inspectable."""
from collections.abc import Mapping
from datetime import datetime
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re

import numpy as np
from .arrays import array_identity, compare_array_maps, compare_nested

SCHEMA = 'jepa4r-gate-record-v1'
REQUIRED = ('gate', 'case_id', 'source_revision', 'producer_sha256', 'input_sha256',
            'config_sha256', 'weight_sha256', 'precision', 'seed', 'counts',
            'actual_exit_code', 'started_at', 'finished_at')
DIGESTS = ('producer_sha256', 'input_sha256', 'config_sha256', 'weight_sha256')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def _structure_errors(record):
    errors = ['missing:' + key for key in REQUIRED if key not in record]
    if errors: return errors
    for key in DIGESTS:
        value = record[key]
        if key == 'weight_sha256' and value is None: continue  # Explicitly not applicable (e.g. A1).
        if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
            errors.append('invalid_sha256:' + key)
    for key in ('gate', 'case_id', 'source_revision', 'precision'):
        if not isinstance(record[key], str) or not record[key]: errors.append('invalid:' + key)
    for key in ('seed', 'actual_exit_code'):
        if not isinstance(record[key], Integral) or isinstance(record[key], bool): errors.append('invalid:' + key)
    counts = record['counts']
    if not isinstance(counts, dict) or set(counts) != {'actual', 'expected'}:
        errors.append('invalid:counts')
    def counts_valid(value):
        if isinstance(value, dict): return bool(value) and all(counts_valid(v) for v in value.values())
        return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0
    if isinstance(counts, dict) and not all(counts_valid(v) for v in counts.values()): errors.append('invalid:count_values')
    try:
        start, end = (datetime.fromisoformat(record[k]) for k in ('started_at', 'finished_at'))
        if start.tzinfo is None or end.tzinfo is None or end < start: raise ValueError()
    except (ValueError, TypeError): errors.append('invalid:timestamps')
    return errors


def _array_metadata(arrays):
    if not isinstance(arrays, dict) or not all(isinstance(k, str) and k for k in arrays):
        raise ValueError('arrays must have nonempty string names')
    return {name: dict(dtype=str(np.asarray(value).dtype), shape=list(np.asarray(value).shape),
                       sha256=array_identity(value)) for name, value in arrays.items()}


def write_record(directory, metadata, arrays):
    """Write a new case only. Nonzero exits/count failures are retained, not approved."""
    if set(metadata) & {'schema', 'arrays', 'arrays_file_sha256'}:
        raise ValueError('reserved record fields must not be supplied by metadata')
    errors = _structure_errors(metadata)
    if errors: raise ValueError('; '.join(errors))
    array_meta = _array_metadata(arrays)
    record = dict(metadata, schema=SCHEMA, arrays=array_meta)
    # Validate JSON before making an output directory; never serialize object/repr state.
    json.dumps(record, allow_nan=False)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / 'arrays.npz').open('xb') as stream: np.savez(stream, **arrays)
    record['arrays_file_sha256'] = file_sha256(directory / 'arrays.npz')
    with (directory / 'record.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return record


def load_record(directory):
    """Verify byte and array identity before handing evidence to a comparator."""
    directory = Path(directory)
    record = json.loads((directory / 'record.json').read_text(encoding='utf-8'))
    errors = _structure_errors(record)
    if record.get('schema') != SCHEMA: errors.append('invalid:schema')
    if errors: raise ValueError('; '.join(errors))
    if record.get('arrays_file_sha256') != file_sha256(directory / 'arrays.npz'):
        raise ValueError('arrays file SHA256 differs')
    with np.load(directory / 'arrays.npz', allow_pickle=False) as packed:
        if len(packed.files) != len(set(packed.files)): raise ValueError('duplicate NPZ names')
        arrays = {name: packed[name] for name in packed.files}
    if record.get('arrays') != _array_metadata(arrays): raise ValueError('array metadata differs')
    return record, arrays


def validate_record(record, arrays):
    errors = _structure_errors(record)
    if record.get('schema') != SCHEMA: errors.append('invalid:schema')
    if record.get('arrays') != _array_metadata(arrays): errors.append('array_metadata_mismatch')
    if record.get('actual_exit_code') != 0: errors.append('producer_nonzero_exit')
    if not arrays: errors.append('empty_array_evidence')
    for name, value in arrays.items():
        value = np.asarray(value)
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            errors.append('nonfinite_or_nonnumeric_array:' + name)
    counts = record.get('counts')
    counts = counts if isinstance(counts, dict) else {}
    if not compare_nested(counts.get('actual'), counts.get('expected'))['passed']: errors.append('count_mismatch')
    return dict(passed=not errors, errors=errors)


def compare_records(actual_directory, reference_directory, *, expected_actual, expected_reference, **options):
    actual, actual_arrays = load_record(actual_directory)
    reference, reference_arrays = load_record(reference_directory)
    checks = dict(actual_record=validate_record(actual, actual_arrays),
                  reference_record=validate_record(reference, reference_arrays))
    for role, record, expected in [('actual', actual, expected_actual), ('reference', reference, expected_reference)]:
        trusted = ('source_revision', 'producer_sha256')
        if (not isinstance(expected, Mapping) or any(key not in expected for key in trusted)
                or not isinstance(expected.get('source_revision'), str) or not expected['source_revision']
                or not isinstance(expected.get('producer_sha256'), str)
                or re.fullmatch('[0-9a-f]{64}', expected['producer_sha256']) is None):
            checks[role + '_bindings'] = dict(passed=False, reason='missing_or_invalid_trusted_source_and_producer')
        else:
            checks[role + '_bindings'] = compare_nested({k: record.get(k) for k in expected}, expected)
    semantic = ('gate', 'input_sha256', 'config_sha256', 'weight_sha256', 'precision', 'seed')
    checks['semantic_bindings'] = compare_nested({k: actual[k] for k in semantic}, {k: reference[k] for k in semantic})
    checks['arrays'] = compare_array_maps(actual_arrays, reference_arrays, **options)
    return dict(passed=all(c['passed'] for c in checks.values()), checks=checks)
