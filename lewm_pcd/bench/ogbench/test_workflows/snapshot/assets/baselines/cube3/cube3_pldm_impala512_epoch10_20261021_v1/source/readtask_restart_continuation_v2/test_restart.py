"""Only the changed epoch3 metadata and native acceptance seams."""
import ast
import importlib.util
from pathlib import Path
import sys
import pytest


def api():
    path = Path(__file__).with_name('continuation.py')
    spec = importlib.util.spec_from_file_location('restart_continuation_control', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unchanged_training_budget_and_parent3():
    module = api()
    old = module.api()
    p = dict(name='A1_native20_pf2', read_batch_size=None, train_workers=20,
             train_prefetch=2, validation_workers=20, validation_prefetch=2, split_validation=False)
    value = old.protocol(p)
    assert (value['parent_updates'], value['new_updates'], value['total_updates'],
            value['new_validation_batches'], value['batch_size'], value['epochs']) == (137109,319921,457030,31997,128,10)
    source = (module.HERE.parent/'readtask_continuation_v1/continuation.py').read_text()
    text = ast.unparse(module.epoch3_tree(source))
    assert 'trainer.current_epoch != 3' in text
    assert 'trainer.current_epoch != 2' not in text
    assert 'torch' not in sys.modules


def test_native_acceptance_keeps_math_and_final_counter_checks():
    module = api()
    source = (module.HERE.parent/'train.py').read_text()
    text = ast.unparse(module.api().acceptance_tree(source))
    assert 'epochs[3:]' in text and 'set(paths[3:])' in text
    assert "summary['metrics']['validate/loss_epoch']['count'] < 7" in text
    assert "full.get('global_step') != expected" in text
    assert 'set(steps) != {expected}' in text
    assert 'strict=True' in text


def test_three_actual_inherited_exports_and_reject_mismatch():
    module = api()
    rows = [dict(epoch=i, path=str(i), sha256=str(i)) for i in range(1,11)]
    parents = [dict(r, origin='actual_original_'+str(r['epoch'])) for r in rows[:3]]
    value = module.lineage(rows, parents)
    assert [r['origin'] for r in value[:3]] == [r['origin'] for r in parents]
    assert all(r['origin']=='produced_by_this_run' for r in value[3:])
    with pytest.raises(ValueError):
        module.lineage([dict(r, sha256='changed') if r['epoch']==3 else r for r in rows], parents)
