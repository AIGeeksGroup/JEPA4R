"""Explicit relocation descriptor for the retained Cube1 JEPA4R smoke export."""
import json
from pathlib import Path


def create_descriptor(output, *, original_receipt, checkpoint, factory_config, baseline_root,
                      compatibility_evidence, adapter, preprocessing_source):
    from .workflow_downstream import sha
    def bind(path):
        path = Path(path).resolve()
        return dict(path=str(path), sha256=sha(path))
    descriptor = dict(schema='cube_retained_encoder_import_v1', scope='retained_export_no_new_training',
        source_revision='f437d1a3f27c3cb1f9dc1d7adfbf0afb68dbaa16', model='lewm_pcd', cube_count=1, mode='smoke',
        original_receipt=bind(original_receipt), checkpoint=bind(checkpoint), factory_config=bind(factory_config),
        baseline_manifest=bind(Path(baseline_root)/'SOURCE_MANIFEST.json'),
        compatibility_evidence=bind(compatibility_evidence), adapter=bind(adapter),
        preprocessing_source=bind(preprocessing_source))
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        json.dump(descriptor, stream, indent=2)
    return bind(output)


def validate_import(record, config):
    """Validate originals by content; never change their paths or PASS fields."""
    from .workflow_downstream import pinned, read, sha
    def same_pin(actual, expected):
        return (isinstance(actual, dict) and actual.get('sha256') == expected['sha256'] and
                pinned(actual).resolve() == pinned(expected).resolve())
    desc = read(pinned(record))
    expected = dict(schema='cube_retained_encoder_import_v1', scope='retained_export_no_new_training',
        source_revision='f437d1a3f27c3cb1f9dc1d7adfbf0afb68dbaa16', model='lewm_pcd', cube_count=1, mode='smoke')
    if any(desc.get(k) != value for k, value in expected.items()) or any(
            config.get(k) != expected[k] for k in ('model', 'cube_count', 'mode')):
        raise ValueError('retained import only accepts the explicit Cube1 JEPA4R smoke case')
    if not same_pin(config.get('encoder_receipt'), desc['original_receipt']):
        raise ValueError('retained import identifies another original receipt')
    old = read(pinned(desc['original_receipt']))
    if (old.get('schema') != 'cube_fresh_wm_acceptance_v1' or
            type(old.get('actual_exit_code')) is not int or old['actual_exit_code'] != 0 or
            old.get('strict_reload_verified') is not True or
            any(old.get(k) != expected[k] for k in ('model', 'cube_count', 'mode'))):
        raise ValueError('original retained producer identity/strict reload receipt differs')
    checkpoint, factory = pinned(desc['checkpoint']).resolve(), pinned(desc['factory_config']).resolve()
    if (old['checkpoint']['sha256'] != desc['checkpoint']['sha256'] or
            old['config']['sha256'] != desc['factory_config']['sha256'] or
            factory != checkpoint.parent/'config.json'):
        raise ValueError('retained weights/factory bytes differ from their original producer')
    binding = config['encoder_binding']
    if (Path(binding.get('checkpoint', '')).resolve() != checkpoint or
            binding.get('checkpoint_sha256') != desc['checkpoint']['sha256'] or
            not same_pin(binding.get('factory_config'), desc['factory_config']) or binding.get('family') != 'lewm' or
            binding.get('latent_dim') != 192 or binding.get('epoch') != old.get('actual_epochs')):
        raise ValueError('resolved encoder binding differs from the retained import')
    model_config = read(factory)
    encoder = model_config.get('encoder', {})
    if (model_config.get('_target_') != 'jepa.JEPA' or 'DexArtPointNet' not in encoder.get('_target_', '') or
            encoder.get('in_dim') != 6 or encoder.get('out_dim') != 192 or
            model_config.get('action_encoder', {}).get('input_dim') != 25):
        raise ValueError('retained model factory identity/dimensions differ')
    baseline_path = pinned(desc['baseline_manifest']).resolve()
    baseline = read(baseline_path)
    if baseline.get('schema') != 'cube_paper_release_sources_v1':
        raise ValueError('retained import requires its frozen old source manifest')
    available = {sha(baseline_path)}
    for name, expected_sha in baseline['files'].items():
        path = (baseline_path.parent/name).resolve()
        if not path.is_relative_to(baseline_path.parent) or sha(path) != expected_sha:
            raise ValueError('frozen old producer source checksum differs: ' + name)
        available.add(expected_sha)
    if any(value not in available for value in old.get('source_binding', {}).values()):
        raise ValueError('original training source binding is absent from the frozen old source')
    if not same_pin(binding.get('adapter'), desc['adapter']):
        raise ValueError('retained encoder adapter binding differs')
    pinned(desc['adapter'])
    pre = binding['preprocessing']
    if (not same_pin(dict(path=pre.get('source_path'), sha256=pre.get('source_sha256')), desc['preprocessing_source']) or
            any(pre.get(k) != value for k, value in dict(device='cpu', input_dtype='float32',
                layout='XYZRGB1200', center=[.425,0.,.2], scale=4., use_rgb=True).items())):
        raise ValueError('retained preprocessing identity or constants differ')
    pinned(desc['preprocessing_source'])
    if (desc['adapter']['sha256'] != baseline['files']['lewm_pcd/bench/ogbench/workflow_lewm_encoder.py'] or
            desc['preprocessing_source']['sha256'] != baseline['files']['lewm_pcd/utils.py']):
        raise ValueError('relocated adapter/preprocessing code differs from the frozen producer')
    evidence = read(pinned(desc['compatibility_evidence']))
    if (evidence.get('actual_exit_code') != 0 or evidence.get('comparison', {}).get('passed') is not True or
            evidence.get('case', {}).get('model') != 'JEPA4R' or evidence['case'].get('cube_count') != 1 or
            evidence.get('identities', {}).get('checkpoint_sha256') != desc['checkpoint']['sha256']):
        raise ValueError('retained import lacks matching fixed-case compatibility evidence')
    return old
