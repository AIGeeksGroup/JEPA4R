"""Paper-only dispatch, state and no-duplicate guards without real children."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def api():
    path = Path(__file__).resolve().parents[1] / 'workflow.py'
    assert path.is_file(), 'paper workflow controller is missing'
    spec = importlib.util.spec_from_file_location('_cube_workflow_controller_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # This suite isolates dispatch; the assembled release hash gate is separate.
    module.verify_sources = lambda: {'unit_test_stub': True}
    return module


def config(tmp_path, model='lewm_pcd', mode='smoke'):
    value = dict(schema='cube_workflow_v1', cube_count=4, model=model, mode=mode,
                 output_root=str(tmp_path/'run'),
                 python=dict(world_model=sys.executable, downstream=sys.executable),
                 workers=1, prefetch=2)
    path = tmp_path/'run-config.json'
    path.write_text(json.dumps(value))
    return path, value


@pytest.mark.parametrize('model', ('lewm_rgb','lewm_pcd','pldm','lejepa','visreg','vjepa'))
def test_six_models_plan_without_frameworks_or_outputs(tmp_path, model):
    module = api()
    path, value = config(tmp_path, model)
    plan = module.build_plan(path)
    stages = [row['stage'] for row in plan['stages']]
    assert 'encode' in stages and stages[-1] == 'native-audit'
    assert ('train' in stages) == (model not in ('visreg','vjepa'))
    assert 'gciql' not in stages and 'accept' not in stages
    assert not Path(value['output_root']).exists()
    assert not any(name in module.__dict__ for name in ('torch','jax','mujoco'))


def test_formal_plan_has_full_policy_train_and_accept(tmp_path):
    module = api()
    path, _ = config(tmp_path, mode='formal')
    stages = [row['stage'] for row in module.build_plan(path)['stages']]
    assert stages.index('reload') < stages.index('gciql') < stages.index('accept') < stages.index('native-sr')


def test_collection_exposes_the_declared_seed_flag(tmp_path):
    module = api()
    path, _ = config(tmp_path)
    command = module.build_plan(path, 'collect')['stages'][0]['command']
    assert command[command.index('--collection-seed')+1] == '0'


@pytest.mark.parametrize('extra', ('encoder_receipt','encoder_binding','camera'))
def test_unrequested_external_recipe_fields_are_rejected(tmp_path, extra):
    module = api()
    path, cfg = config(tmp_path)
    cfg[extra] = str(tmp_path/'external.json')
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='unknown'):
        module.build_plan(path)


def test_collection_seed_mismatch_is_rejected_before_collection(tmp_path):
    module = api()
    path, cfg = config(tmp_path)
    cfg['collection_seed'] = 7
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match='seed'):
        module.build_plan(path)


def test_smoke_and_formal_default_data_namespaces_are_separate(tmp_path):
    module = api()
    path, cfg = config(tmp_path)
    smoke = module.build_plan(path)['config']
    cfg.update(mode='formal',output_root=str(tmp_path/'formal-run'))
    path.write_text(json.dumps(cfg))
    formal = module.build_plan(path)['config']
    assert smoke['collection_root'] != formal['collection_root']
    assert smoke['data_root'] != formal['data_root']
    assert smoke['official_root'] == formal['official_root']


def test_direct_switch_builds_a_plan_without_a_config_file(tmp_path, capsys):
    module = api()
    output = tmp_path/'not-created'
    assert module.main(['--model','jepa4r','--cube-count','2','--mode','smoke',
                        '--output',str(output),'--wm-python',sys.executable,
                        '--rl-python',sys.executable]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value['config']['model'] == 'lewm_pcd'
    assert value['config']['cube_count'] == 2
    assert not output.exists()


def test_camera_stage_assembles_a_verified_runtime_before_probe(tmp_path):
    module = api()
    path, _ = config(tmp_path)
    command = module.build_plan(path,'camera')['stages'][0]['command']
    assert command[command.index('--internal')+1] == 'camera'


def test_foundation_encoder_does_not_have_a_world_model_train_stage(tmp_path):
    module = api()
    path, _ = config(tmp_path,'vjepa')
    with pytest.raises(ValueError,match='released'):
        module.build_plan(path,'train')


@pytest.mark.parametrize('key,value', [('cube_count',0),('model','unknown_model'),('mode','debug')])
def test_unknown_group_or_mode_is_rejected(tmp_path, key, value):
    module = api()
    path, cfg = config(tmp_path)
    cfg[key] = value
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        module.build_plan(path)


def test_unknown_config_key_is_rejected(tmp_path):
    module = api()
    path, cfg = config(tmp_path)
    cfg['unknown_option'] = 'unsupported'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='unknown'):
        module.build_plan(path)


def test_child_failure_persists_real_exit_without_success_receipt(tmp_path, monkeypatch):
    module = api()
    path, cfg = config(tmp_path)
    plan = module.build_plan(path, 'collect')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=7))
    assert module.execute(plan) == 7
    root = Path(cfg['output_root'])
    state = json.loads((root/'STATUS.json').read_text())
    assert state['stage'] == 'failed' and state['actual_exit_code'] == 7
    assert state['failed_count'] == 1 and state['completed_count'] == 0
    assert not (root/'PIPELINE_COMPLETE.json').exists()
    assert (root/'dispatch/collect/stdout_stderr.log').is_file()


def test_exit_zero_without_required_output_is_a_failure(tmp_path, monkeypatch):
    module = api()
    path, cfg = config(tmp_path)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    assert module.execute(module.build_plan(path, 'collect')) != 0
    assert not (Path(cfg['output_root'])/'PIPELINE_COMPLETE.json').exists()


def test_executed_stage_requires_explicit_zero_exit_in_its_receipt(tmp_path, monkeypatch):
    module = api()
    path, cfg = config(tmp_path)
    plan = module.build_plan(path,'collect')
    def child(*args, **kwargs):
        target = Path(plan['stages'][0]['receipt'])
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({'cube_count':4,'mode':'smoke'}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',child)
    assert module.execute(plan) != 0
    assert not (Path(cfg['output_root'])/'PIPELINE_COMPLETE.json').exists()


def test_same_stage_cannot_silently_dispatch_twice(tmp_path, monkeypatch):
    module = api()
    path, _ = config(tmp_path)
    plan = module.build_plan(path, 'collect')
    calls = []

    def child(*args, **kwargs):
        calls.append(args)
        target = Path(plan['stages'][0]['receipt'])
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({'actual_exit_code':0,'cube_count':4,'mode':'smoke'}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, 'run', child)
    assert module.execute(plan) == 0
    with pytest.raises(FileExistsError):
        module.execute(plan)
    assert len(calls) == 1


def test_complete_receipt_requires_the_entire_chain(tmp_path, monkeypatch):
    module = api()
    path, cfg = config(tmp_path)
    plan = module.build_plan(path, 'collect')

    def child(*args, **kwargs):
        target = Path(plan['stages'][0]['receipt'])
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({'actual_exit_code':0,'cube_count':4,'mode':'smoke'}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, 'run', child)
    assert module.execute(plan) == 0
    assert not (Path(cfg['output_root'])/'PIPELINE_COMPLETE.json').exists()


def completed_gate(tmp_path, monkeypatch, stage='gate'):
    """A real file transaction behind the existing gate-stage receipt schema."""
    module = api()
    path, cfg = config(tmp_path)
    plan = module.build_plan(path, stage)
    receipt = Path(plan['stages'][0]['receipt'])
    gate = Path(cfg['output_root'])/'downstream/gate'
    checkpoint = gate/'checkpoints/update_000000002'
    files = dict(params=checkpoint/'params_2.pkl', state=checkpoint/'run_state.pkl',
                 full_state=gate/'full_state.npz', actions=gate/'next_actions.npy')
    calls = []

    def child(*args, **kwargs):
        calls.append(args)
        checkpoint.mkdir(parents=True)
        for name, target in files.items():
            target.write_bytes(('unchanged '+name).encode())
        (checkpoint/'manifest.json').write_text(json.dumps(dict(counter=2,
            agent_sha256=module.sha(files['params']), sidecar_sha256=module.sha(files['state']))))
        report = gate/'report.json'
        report.write_text(json.dumps(dict(actual_exit_code=0, updates=2, checkpoint=str(checkpoint),
            full_state_sha256=module.sha(files['full_state']),
            action_reference_sha256=module.sha(files['actions']))))
        if stage == 'reload':
            report = gate.parent/'reload.json'
            report.write_text(json.dumps(dict(actual_exit_code=0, backend='cpu',
                action_reference_sha256=module.sha(files['actions']),
                same_cpu_action_consistency_verified=False)))
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps(dict(actual_exit_code=0, cube_count=4, mode='smoke',
            artifacts=dict(gate_report=dict(path=str(report), sha256=module.sha(report))))))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, 'run', child)
    assert module.execute(plan) == 0
    return module, plan, cfg, files, calls


def rejected_resume(module, plan):
    try:
        result = module.execute(plan, resume=True)
    except (ValueError, FileNotFoundError):
        return
    assert result != 0, 'resume accepted changed or missing completed-stage evidence'


def test_completed_resume_checks_real_artifacts_without_redispatch(tmp_path, monkeypatch):
    module, plan, cfg, files, calls = completed_gate(tmp_path, monkeypatch)
    receipt = Path(plan['stages'][0]['receipt']).read_bytes()
    assert module.execute(plan, resume=True) == 0
    assert len(calls) == 1
    assert Path(plan['stages'][0]['receipt']).read_bytes() == receipt
    assert all(target.is_file() for target in files.values())


def test_completed_resume_keeps_legacy_reload_reference_location(tmp_path, monkeypatch):
    module, plan, cfg, files, calls = completed_gate(tmp_path, monkeypatch, stage='reload')
    assert module.execute(plan, resume=True) == 0
    assert len(calls) == 1


@pytest.mark.parametrize('target', ('report', 'params', 'state', 'full_state', 'actions'))
def test_completed_resume_rejects_direct_and_nested_artifact_corruption(tmp_path, monkeypatch, target):
    module, plan, cfg, files, calls = completed_gate(tmp_path, monkeypatch)
    artifact = Path(cfg['output_root'])/'downstream/gate/report.json' if target == 'report' else files[target]
    artifact.write_bytes(b'corrupted actual artifact')
    rejected_resume(module, plan)
    assert len(calls) == 1


@pytest.mark.parametrize('evidence', ('nonzero_exit', 'missing_exit', 'missing_completion'))
def test_completed_resume_rejects_missing_or_failed_dispatch_evidence(tmp_path, monkeypatch, evidence):
    module, plan, cfg, files, calls = completed_gate(tmp_path, monkeypatch)
    driver = Path(cfg['output_root'])/'dispatch/gate'
    if evidence == 'nonzero_exit':
        (driver/'actual_exit_code').write_text('7\n')
    elif evidence == 'missing_exit':
        (driver/'actual_exit_code').unlink()
    else:
        (driver/'COMPLETED.json').unlink()
    rejected_resume(module, plan)
    assert len(calls) == 1


def test_completed_resume_rejects_redirected_canonical_receipt(tmp_path, monkeypatch):
    module, plan, cfg, files, calls = completed_gate(tmp_path, monkeypatch)
    root = Path(cfg['output_root'])
    original = Path(plan['stages'][0]['receipt'])
    other = root/'foreign-receipt.json'
    other.write_bytes(original.read_bytes())
    state = json.loads((root/'STATUS.json').read_text())
    state['completed']['gate']['path'] = str(other)
    (root/'STATUS.json').write_text(json.dumps(state))
    rejected_resume(module, plan)
    assert len(calls) == 1


def test_completed_collection_resume_accepts_successful_retry_and_rechecks_input(tmp_path, monkeypatch):
    module = api()
    path, cfg = config(tmp_path)
    plan = module.build_plan(path, 'collect')
    target = Path(plan['stages'][0]['receipt'])
    data = target.parent/'raw.h5'
    calls = []

    def child(*args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            return SimpleNamespace(returncode=7)
        target.parent.mkdir(parents=True)
        data.write_bytes(b'accepted raw input bytes')
        target.write_text(json.dumps(dict(actual_exit_code=0, cube_count=4, mode='smoke',
            producer_binding=dict(files={}), data_files={str(data):dict(sha256=module.sha(data))})))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, 'run', child)
    assert module.execute(plan) == 7
    assert module.execute(plan, resume=True) == 0
    root = Path(cfg['output_root'])
    assert (root/'dispatch/collect/actual_exit_code').read_text().strip() == '7'
    assert (root/'dispatch/collect/attempts/attempt_001/actual_exit_code').read_text().strip() == '0'
    assert module.execute(plan, resume=True) == 0
    assert len(calls) == 2
    data.write_bytes(b'changed actual raw input')
    rejected_resume(module, plan)
    assert len(calls) == 2
    assert (root/'dispatch/collect/actual_exit_code').read_text().strip() == '7'


def completed_cache(tmp_path, monkeypatch):
    module = api()
    path, cfg = config(tmp_path)
    plan = module.build_plan(path, 'encode')
    receipt = Path(plan['stages'][0]['receipt'])
    cache = Path(cfg['output_root'])/'downstream/cache'
    latent = cache/'train/episode_000000/latent.npy'
    calls = []

    def child(*args, **kwargs):
        calls.append(args)
        latent.parent.mkdir(parents=True)
        latent.write_bytes(b'accepted actual encoder features')
        episode = latent.parent/'receipt.json'
        episode.write_text(json.dumps(dict(actual_exit_code=0, split='train', episode=0,
            latent=dict(path=str(latent), sha256=module.sha(latent)))))
        manifest = cache/'cache_manifest.json'
        manifest.write_text(json.dumps(dict(actual_exit_code=0,
            splits=dict(train=[dict(episode=0, receipt_path=str(episode),
                                    receipt_sha256=module.sha(episode))]))))
        receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps(dict(actual_exit_code=0, cube_count=4, mode='smoke',
            artifacts=dict(cache_manifest=dict(path=str(manifest),sha256=module.sha(manifest))))))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, 'run', child)
    assert module.execute(plan) == 0
    return module, plan, latent, calls


def test_completed_resume_checks_nested_cache_without_redispatch(tmp_path, monkeypatch):
    module, plan, latent, calls = completed_cache(tmp_path, monkeypatch)
    assert module.execute(plan, resume=True) == 0
    assert len(calls) == 1


def test_completed_resume_rejects_nested_cache_latent_corruption(tmp_path, monkeypatch):
    module, plan, latent, calls = completed_cache(tmp_path, monkeypatch)
    latent.write_bytes(b'changed actual encoder features')
    rejected_resume(module, plan)
    assert len(calls) == 1
