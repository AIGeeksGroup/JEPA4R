"""Training-camera delta only over the frozen Quad 2D materializer v1."""
import copy
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

V1=Path(__file__).resolve().parent.parent/'adapter_v1/materialize.py'
assert hashlib.sha256(V1.read_bytes()).hexdigest()=='f0ec88f369936a319ead35063a87d51935234bceca4ef6714a013b400fbf34c5'
spec=importlib.util.spec_from_file_location('_frozen_quad_2d_materializer_v1',V1)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
MODULE=base.MODULE
SOURCE_POSITION=[1.287,0.,.509]
TRAINING_POSITION=[1.3835957918490516,1.1102230246251566e-17,.5348699245660802]
V1_EFFECTIVE={
    MODULE+'/produce.py':'d8b5bdb45c5840ef58fd845fef858963c804dd4b265d0d6a59d29c5b4fdfb68a',
    MODULE+'/render_adapter.py':'baf424c2f5ff58708764ce53e723e43be3589830ea639720d5fe526abcd45ac4',
    MODULE+'/audit_cache.py':'d67fc2d20fa5699ff02fc9c4463a4dbceb9618f4f69619c3913b60275b87b8de',
}
ORIGINAL_BUILD_SOURCES=base.build_sources
ORIGINAL_MAKE_PLAN=base.make_plan


def build_sources(original):
    sources,ledger=ORIGINAL_BUILD_SOURCES(original)
    for name,expected in V1_EFFECTIVE.items():
        assert hashlib.sha256(sources[name]).hexdigest()==expected,name
    def edit(name,changes):
        before=sources[name];text=before.decode()
        for old,new,count in changes:
            assert text.count(old)==count,(name,old,count,text.count(old))
            text=text.replace(old,new)
        base.ast.parse(text);sources[name]=text.encode()
        record=next(item for item in ledger if item['path']==name)
        record['camera_v2_delta']=dict(original_effective_v1_sha256=V1_EFFECTIVE[name],changes=changes)
        record['replacements']+=changes
        record['generated_sha256']=hashlib.sha256(sources[name]).hexdigest()
        record['diff']=''.join(difflib.unified_diff((original/name).read_text().splitlines(True),text.splitlines(True),
            fromfile='accepted_cube2/'+name,tofile='new_cube4_training_camera_v2/'+name))
    edit(MODULE+'/render_adapter.py',[
        ("np.testing.assert_allclose(position, camera['position'], atol=1e-12, rtol=0)",
         "np.testing.assert_allclose(position, camera['source_position'], atol=1e-12, rtol=0)",1),
        ("    # Match approved collection's front_pixels optical alias of original front.\n",
         "    camera_rotation = np.empty(9)\n"
         "    mujoco.mju_quat2Mat(camera_rotation, actual)\n"
         "    render_position = position + .10 * camera_rotation.reshape(3, 3)[:, 2]\n"
         "    np.testing.assert_allclose(render_position, camera['position'], atol=1e-12, rtol=0)\n"
         "    # Match the training camera: same +5deg pose, local +Z 0.10m pullback.\n",1),
        ('= position, actual, 45.','= render_position, actual, 45.',1),
        ('pullback=0, crop=None','pullback=.10, crop=None',1)])
    edit(MODULE+'/audit_cache.py',[
        ('pullback=0, crop=None','pullback=.10, crop=None',1),
        ("    for key in ('source_position', 'position'):\n"
         "        np.testing.assert_allclose(actual[key], [1.287, 0., .509], rtol=0., atol=1e-12)\n",
         "    for key, expected in (('source_position', [1.287, 0., .509]),\n"
         "            ('position', [1.3835957918490516, 1.1102230246251566e-17, .5348699245660802])):\n"
         "        np.testing.assert_allclose(actual[key], expected, rtol=0., atol=1e-12)\n"
         "        np.testing.assert_allclose(planned[key], expected, rtol=0., atol=1e-12)\n",1)])
    assert hashlib.sha256(sources[MODULE+'/produce.py']).hexdigest()==V1_EFFECTIVE[MODULE+'/produce.py']
    return sources,ledger


def make_plan(original,case,root,binding,sources):
    reference=json.loads((original/MODULE/'producer_plan.json').read_text())['camera']
    expected=copy.deepcopy(reference)
    expected.update(source_position=SOURCE_POSITION,position=TRAINING_POSITION,pullback=.10)
    assert binding['camera']==expected,'Only the approved training front+5/local+Z0.10m/fullFOV camera'
    # Reuse all v1 data/checkpoint/source/path checks. Its old-camera equality
    # checks the unchanged template; the actual new camera is strictly checked
    # above and placed in the returned plan before it can be written.
    template_binding=copy.deepcopy(binding);template_binding['camera']=reference
    plan=ORIGINAL_MAKE_PLAN(original,case,root,template_binding,sources)
    plan['camera']=copy.deepcopy(binding['camera'])
    return plan


if __name__=='__main__':
    base.build_sources=build_sources
    base.make_plan=make_plan
    base.main()
