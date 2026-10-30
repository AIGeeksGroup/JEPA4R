"""Only stdlib contracts; no data, Torch, GPU or live process access."""
import ast
import importlib.util
from pathlib import Path
import pytest

HERE=Path(__file__).parent


def api():
    path=HERE/'continuation.py'
    assert path.exists(), 'missing isolated continuation entry'
    spec=importlib.util.spec_from_file_location('pldm_continuation_tests',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_resource_only_worker_switch_and_total_budget():
    m=api()
    assert m.protocol(20)==dict(train_workers=20,validation_workers=20,batch_size=128,
        precision='bf16-mixed',epochs=10,parent_updates=45703,new_updates=411327,
        total_updates=457030,train_batches=45703,validation_batches=4571,validation_tail=40)
    assert m.protocol(24)=={**m.protocol(20),'train_workers':24}
    with pytest.raises(ValueError):m.protocol(32)


def test_actual_selection_is_recomputed_not_trusted_label():
    m=api()
    class Trial:
        CASES=[('A1_native20',20),('B_native24',24),('A2_native20',20)]
        @staticmethod
        def compare_cohort(rows):
            if any(r['ids']!=rows[0]['ids'] for r in rows):raise ValueError('IDs differ')
        @staticmethod
        def recommend_workers(rates):return {'recommended_train_workers':24 if rates[1]>max(rates[0],rates[2])*1.05 else 20}
    rows={name:dict(actual_exit_code=0,ids=[1,3],updates_per_second=8 if w==24 else 7)
          for name,w in Trial.CASES}
    report=dict(stage='worker_ABA_complete_awaiting_Root_formal_release',exit_code=0,
        all_diagnostic_updates_discarded=True,recommended_train_workers=24,cases=rows)
    assert m.selected_workers(report,Trial)==24
    with pytest.raises(ValueError):m.selected_workers({**report,'recommended_train_workers':20},Trial)
    bad={**rows,'B_native24':{**rows['B_native24'],'ids':[3,1]}}
    with pytest.raises(ValueError):m.selected_workers({**report,'cases':bad},Trial)
    with pytest.raises(ValueError):m.selected_workers({**report,'exit_code':1},Trial)


def test_only_parent_export_and_nine_new_validation_guards_change():
    m=api();source=(HERE.parent/'train.py').read_text(encoding='utf8')
    original=ast.parse(source);changed=m.acceptance_tree(source)
    old={n.name:ast.dump(n) for n in original.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    new={n.name:ast.dump(n) for n in changed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert [k for k in old if old[k]!=new[k]]==['accept']
    text=ast.unparse(changed)
    assert "paths = [Path(CONTINUATION_PARENT_BARE)]" in text
    assert "set(paths[1:])" in text
    assert "['validate/loss_epoch']['count'] < 9" in text
    assert "expected - 1" in text
    with pytest.raises(ValueError):m.acceptance_tree(ast.unparse(original).replace("set(export.glob('weights_epoch_*.pt')) != set(paths)","False"))


def test_actual_parent_and_produced_exports_have_truthful_lineage():
    m=api();rows=[dict(epoch=i,path='parent' if i==1 else 'new'+str(i)) for i in range(1,11)]
    out=m.lineage(rows,'parent')
    assert out[0]['origin']=='inherited_parent'
    assert all(r['origin']=='produced_by_this_run' for r in out[1:])
    with pytest.raises(ValueError):m.lineage(rows,'wrong')
    with pytest.raises(ValueError):m.lineage(rows[1:],'parent')
