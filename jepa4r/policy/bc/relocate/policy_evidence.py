# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/relocate/policy_evidence.py (verbatim)
"""Real-input references for corresponding in-memory and independently loaded policies."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random

import numpy as np


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def _identity(value):
    a = _array(value)
    return hashlib.sha256((str(a.dtype) + str(a.shape)).encode() + a.tobytes()).hexdigest()


def _state(policy):
    state = {name: _array(value) for name, value in policy.model.state_dict().items()}
    if not state or any(value.dtype.kind not in 'biufc' or not np.isfinite(value).all()
                        for value in state.values()):
        raise ValueError('policy state must contain finite numeric arrays')
    return state


@contextmanager
def _preserve_rng():
    py, numpy = random.getstate(), np.random.get_state()
    try:
        import torch
    except ImportError:
        torch = None
    cpu = torch.get_rng_state() if torch is not None else None
    cuda = torch.cuda.get_rng_state_all() if torch is not None and torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(numpy)
        if torch is not None:
            torch.set_rng_state(cpu)
            if cuda is not None:
                torch.cuda.set_rng_state_all(cuda)


def policy_actions(policy, features):
    x = _array(features)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('real policy features must be a nonempty finite matrix')
    if getattr(policy.model, 'training', False):
        raise ValueError('policy evidence requires the corresponding eval-mode policy')
    if getattr(policy, 'n', x.shape[1]) != x.shape[1]:
        raise ValueError('real policy feature dimension differs')
    before = {name: _identity(value) for name, value in _state(policy).items()}
    with _preserve_rng():
        actions = np.stack([_array(policy.get_action(row.copy())[1]['evaluation']) for row in x])
    if actions.ndim != 2 or not np.isfinite(actions).all():
        raise ValueError('policy actions must be a finite matrix')
    if getattr(policy, 'm', actions.shape[1]) != actions.shape[1]:
        raise ValueError('real policy action dimension differs')
    if before != {name: _identity(value) for name, value in _state(policy).items()}:
        raise ValueError('policy state changed during evidence capture')
    return actions


def write_policy_reference(directory, label, policy, features, *, epoch, score):
    if not label or Path(label).name != label:
        raise ValueError('policy evidence label must be one path component')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays_path, metadata_path = directory / (label + '.npz'), directory / (label + '.json')
    if arrays_path.exists() or metadata_path.exists():
        raise FileExistsError('policy evidence already exists: ' + label)
    features = _array(features)
    actions = policy_actions(policy, features)
    state = _state(policy)
    if not np.isfinite(float(score)):
        raise ValueError('policy evaluation score must be finite')
    with arrays_path.open('xb') as stream:
        np.savez_compressed(stream, features=features, actions=actions,
                            **{'state.' + name: value for name, value in state.items()})
    metadata = dict(schema='real-policy-reference-v1', epoch=int(epoch), score=float(score),
                    policy_class=type(policy).__module__ + '.' + type(policy).__name__,
                    arrays_file=arrays_path.name,
                    arrays_sha256=hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
                    features_sha256=_identity(features), actions_sha256=_identity(actions),
                    policy_state={name: _identity(value) for name, value in state.items()})
    with metadata_path.open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
    return dict(metadata_path=str(metadata_path.resolve()), arrays_path=str(arrays_path.resolve()),
                epoch=int(epoch), score=float(score))


def verify_policy_reference(policy, metadata_path, *, atol=1e-6, rtol=1e-4):
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata['schema'] != 'real-policy-reference-v1':
        raise ValueError('unsupported policy reference schema')
    if type(policy).__module__ + '.' + type(policy).__name__ != metadata['policy_class']:
        raise ValueError('policy class differs from the corresponding reference')
    relative = Path(metadata['arrays_file'])
    if relative.name != str(relative):
        raise ValueError('policy reference arrays must remain beside metadata')
    path = metadata_path.parent / relative
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['arrays_sha256']:
        raise ValueError('policy evidence payload changed')
    state = {name: _identity(value) for name, value in _state(policy).items()}
    if state != metadata['policy_state']:
        raise ValueError('policy state differs from the corresponding reference epoch')
    with np.load(path, allow_pickle=False) as arrays:
        features, expected = arrays['features'], arrays['actions']
        if _identity(features) != metadata['features_sha256'] or _identity(expected) != metadata['actions_sha256']:
            raise ValueError('policy evidence array identity differs')
        actual = policy_actions(policy, features)
    if actual.shape != expected.shape:
        raise ValueError('policy action shape differs')
    error = np.abs(actual - expected)
    mismatches = int(np.count_nonzero(error > atol + rtol * np.abs(expected)))
    if mismatches:
        raise ValueError('policy actions differ in ' + str(mismatches) + ' elements')
    return dict(passed=True, epoch=metadata['epoch'], feature_shape=list(features.shape),
                action_shape=list(actual.shape), maximum_action_difference=float(error.max()),
                exceeded_elements=mismatches, atol=atol, rtol=rtol,
                metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                arrays_sha256=metadata['arrays_sha256'])


def select_best_reference(records):
    if not records:
        raise ValueError('no actual policy evaluation references')
    return max(enumerate(records), key=lambda pair: (float(pair[1]['score']), pair[0]))[1]


def main(argv=None):
    import argparse
    import pickle
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    with args.policy.open('rb') as stream:
        policy = pickle.load(stream)
    policy.model.to('cpu')
    policy.model.eval()
    result = verify_policy_reference(policy, args.reference)
    result.update(policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(), actual_exit_code=0)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
