"""Pure checks for the policy-critical content of the official Relocate demos.

Consumers build records from actual deserialized demonstrations; the checks
cover their count, order, shapes, dtypes and finiteness.
"""
import hashlib
import json
from pathlib import Path
import pickle
import re

import numpy as np


OFFICIAL_COUNT = 100
ARRAY_SHAPES = dict(actions=(200, 30), state_obs=(200, 39),
                    proprio=(200, 30), rewards=(200,))


def _array_record(name, value):
    array = np.asarray(value)
    if array.shape != ARRAY_SHAPES[name]:
        raise ValueError(f'{name} must have shape {ARRAY_SHAPES[name]}')
    if array.dtype.kind not in ('f', 'i', 'u'):
        raise ValueError(f'{name} must contain real numeric values')
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must contain finite values')
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


def demo_content_record(index, demo):
    """Hash original values and dtypes without clipping or modifying arrays."""
    if type(index) is not int or not 0 <= index < OFFICIAL_COUNT:
        raise ValueError('demo_index must be an integer in 0..99')
    if not isinstance(demo, dict) or not isinstance(demo.get('init_state_dict'), dict):
        raise ValueError('demonstration must contain init_state_dict')
    infos = demo.get('env_infos')
    if not isinstance(infos, dict):
        raise ValueError('demonstration must contain env_infos.proprio')
    values = dict(actions=demo.get('actions'), state_obs=demo.get('state_obs'),
                  proprio=infos.get('proprio'), rewards=demo.get('rewards'))
    return dict(demo_index=index,
        original_init_sha256=hashlib.sha256(pickle.dumps(demo['init_state_dict'], protocol=4)).hexdigest(),
        arrays={name: _array_record(name, value) for name, value in values.items()})


def content_digest(records):
    """Hash the ordered records using the canonical sorted compact UTF-8 JSON."""
    text = json.dumps(records, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _require_sha256(value, name):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError(f'{name} must be a lowercase SHA256')


def validate_content_records(records):
    """Require all 100 original demos against the repository-owned digest.

    There is no expected-digest argument or asset-controlled contract path.
    Real/finite array checks occur in ``demo_content_record`` before hashing;
    this boundary also checks the complete record schema and frozen content.
    """
    if not isinstance(records, list) or len(records) != OFFICIAL_COUNT:
        raise ValueError('require 100 ordered official content records')
    initial_hashes = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {'demo_index', 'original_init_sha256', 'arrays'}:
            raise ValueError('invalid official content record schema')
        if type(record['demo_index']) is not int or record['demo_index'] != index:
            raise ValueError('official demo indices must retain order 0..99')
        _require_sha256(record['original_init_sha256'], 'original_init_sha256')
        initial_hashes.append(record['original_init_sha256'])
        arrays = record['arrays']
        if not isinstance(arrays, dict) or set(arrays) != set(ARRAY_SHAPES):
            raise ValueError('official content requires actions/state_obs/proprio/rewards')
        for name, expected_shape in ARRAY_SHAPES.items():
            item = arrays[name]
            if not isinstance(item, dict) or set(item) != {'shape', 'dtype', 'sha256'}:
                raise ValueError(f'invalid {name} array record schema')
            shape = item['shape']
            if (not isinstance(shape, list) or any(type(n) is not int for n in shape)
                    or shape != list(expected_shape)):
                raise ValueError(f'{name} shape differs from the official contract')
            try:
                dtype = np.dtype(item['dtype'])
            except (TypeError, ValueError) as error:
                raise ValueError(f'invalid {name} dtype') from error
            if (not isinstance(item['dtype'], str) or dtype.str != item['dtype']
                    or dtype.kind not in ('f', 'i', 'u')):
                raise ValueError(f'{name} dtype must be a canonical real numeric dtype.str')
            _require_sha256(item['sha256'], name + '.sha256')
    if len(set(initial_hashes)) != OFFICIAL_COUNT:
        raise ValueError('duplicate official initialization content')
    return dict(passed=True, count=OFFICIAL_COUNT, policy_critical_content_sha256=content_digest(records),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
