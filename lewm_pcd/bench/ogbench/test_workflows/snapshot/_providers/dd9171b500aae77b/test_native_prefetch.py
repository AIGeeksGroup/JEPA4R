"""Only stdlib native-resource/budget/AST/lineage controls; no model/data."""
import ast
import importlib.util
from pathlib import Path
import pytest

HERE=Path(__file__).parent


def api():
    p=HERE/'continuation.py';assert p.exists(),'missing native prefetch continuation bridge'
    s=importlib.util.spec_from_file_location('native_prefetch_bridge_controls',p)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def profile(pf=2):
    return dict(name='synthetic_native',read_batch_size=None,train_workers=20,
        train_prefetch=pf,validation_workers=20,validation_prefetch=2,split_validation=False)


def test_two_native_options_and_six_new_epoch_budget():
    m=api()
    for pf in (2,4):
        p=m.protocol(profile(pf))
        assert (p['parent_updates'],p['new_updates'],p['total_updates'],p['epochs'])==(182812,274218,457030,10)
        assert (p['train_workers'],p['validation_workers'],p['new_validation_batches'],p['validation_tail'])==(20,20,27426,40)
    for wrong in (profile(8),{**profile(),'train_workers':32},{**profile(),'read_batch_size':32},{**profile(),'split_validation':True}):
        with pytest.raises(ValueError):m.protocol(wrong)


def test_only_train_prefetch_changes_no_adapter_or_object_replacement():
    m=api()
    class Loader:
        def __init__(self,train):
            self.batch_size=128;self.num_workers=20;self.prefetch_factor=2
            self.pin_memory=True;self.persistent_workers=False;self.drop_last=train;self.train=train
            self.dataset=object();self.sampler=object();self.batch_sampler=object();self.generator=object()
        def __len__(self):return 45703 if self.train else 4571
    a,b=Loader(True),Loader(False);objects=[(x.dataset,x.sampler,x.batch_sampler,x.generator) for x in (a,b)]
    def forbidden():raise AssertionError('native profile imported split adapter')
    x,y,record=m.install_loaders(a,b,profile(4),forbidden)
    assert x is a and y is b and a.prefetch_factor==4 and b.prefetch_factor==2
    assert objects==[(z.dataset,z.sampler,z.batch_sampler,z.generator) for z in (x,y)]
    assert not record['split_installed'] and record['outer_batch_size']==128


def test_epoch4_ast_only_instrumentation_and_acceptance_boundaries_change():
    m=api();p=HERE.parent/'readtask_continuation_v1/continuation.py';source=p.read_text(encoding='utf8')
    changed=m.native_tree(source);old=ast.parse(source)
    oldf={n.name:ast.dump(n) for n in old.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    newf={n.name:ast.dump(n) for n in changed.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert {k for k in oldf if oldf[k]!=newf[k]}=={'protocol','run_train','acceptance_tree','run_accept','main'}
    text=ast.unparse(changed)
    assert 'trainer.current_epoch != 4' in text and 'epoch_index=4' in text
    assert 'inherited_complete_validation_epochs=4' in text and 'new_complete_validation_epochs=6' in text
    instance=m.api();assert instance.PARENT_UPDATES==182812 and instance.protocol is m.protocol
    native=(HERE.parent/'train.py').read_text(encoding='utf8');tree=instance.acceptance_tree(native)
    original=ast.parse(native);a={n.name:ast.dump(n) for n in original.body if isinstance(n,ast.FunctionDef)}
    b={n.name:ast.dump(n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert [k for k in a if a[k]!=b[k]]==['accept']
    text=ast.unparse(tree);assert 'epochs[4:]' in text and 'set(paths[4:])' in text
    assert "['validate/loss_epoch']['count'] < 6" in text
    assert "val['total']['completed'] != contract(args.method)['validation_batches']" in text


def test_four_real_parent_exports_and_six_new_exports():
    m=api();parents=[dict(epoch=i,path='parent'+str(i),sha256='sha'+str(i),origin='actual_origin'+str(i)) for i in range(1,5)]
    rows=parents+[dict(epoch=i,path='new'+str(i),sha256='sha'+str(i)) for i in range(5,11)]
    result=m.lineage(rows,parents)
    assert [r['origin'] for r in result[:4]]==[r['origin'] for r in parents]
    assert all(r['origin']=='produced_by_this_run' for r in result[4:])
    with pytest.raises(ValueError):m.lineage(rows,parents[:3])
    changed=[dict(r) for r in rows];changed[3]['sha256']='other'
    with pytest.raises(ValueError):m.lineage(changed,parents)
