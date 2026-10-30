"""Register accepted Quad LeWM192 policies over the unchanged M5 Native flow."""
import hashlib
import json
from pathlib import Path
import types
import numpy as np

ROOT = Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'cube23_distance_gciql_v1/native_sr_visual_v1/visual_sr.py'
SOURCE_SHA = 'e40a2ff52bd0915b7953d9011c65e02c75f9e0b2d0e454b613d18c275f6c13f0'
POSITION = [1.3835957918490516, 1.1102230246251566e-17, .5348699245660802]
SOURCE_POSITION = [1.287, 0., .509]
QUATERNION = [.5609588321928806, .4304941214288525, .4304941214288525, .5609588321928807]
CROP = [[-.25, -.5, -.02], [.75, .5, .9]]
EDITS = [("{2: 'double', 3: 'triple'}", "{2: 'double', 3: 'triple', 4: 'quadruple'}"),
    ('{2: "cube-double-play-v0", 3: "cube-triple-play-v0"}',
     '{2: "cube-double-play-v0", 3: "cube-triple-play-v0", 4: "cube-quadruple-play-v0"}')]


def camera_position(quaternion):
    w,x,y,z = np.asarray(quaternion, dtype=np.float64)
    local_z = np.array([2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)])
    return np.array(SOURCE_POSITION)+.1*local_z


def load_core(source=SOURCE, binding_path=None):
    raw = Path(source).read_bytes(); assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
    original = raw.decode(); effective = original
    for before, after in EDITS:
        assert effective.count(before) == 1; effective = effective.replace(before, after)
    restored = effective
    for before, after in reversed(EDITS): restored = restored.replace(after,before)
    assert restored == original
    core = types.ModuleType('_Quad_registered_original_M5'); core.__file__ = str(source)
    exec(compile(effective,str(source),'exec'),core.__dict__)
    core.ROOT = ROOT
    binding = json.loads(Path(binding_path or HERE/'bindings.json').read_text())
    assert binding['actual_exit_code'] == 0 and binding['cube_count'] == 4 and set(binding['groups']) == {'G11','G12'}
    for group, config in binding['groups'].items():
        assert (config['cube_count'],config['state_dim'],config['current_dim'],config['goal_dim']) == (4,55,211,204)
        assert config['mode'] == ('rgb' if group=='G11' else 'pcd')
        assert config['max_episode_steps'] == 1000
        value=dict(config)
        for key in ('wm','training','acceptance','gate','final'): value[key]=Path(value[key])
        core.GROUPS[group]=value
    original_camera=core.configure_camera
    original_camera_for_group=core.camera_for_group
    def camera_for_group(group):
        return dict(original_camera_for_group(group), source_position=SOURCE_POSITION,
            position=POSITION, pullback=.1, crop=None)
    def configure_camera(base,group):
        # Original source-front assertion and local-X5 rotation run first, unchanged.
        result=original_camera(base,group)
        import mujoco
        target=mujoco.mj_name2id(base._model,mujoco.mjtObj.mjOBJ_CAMERA,'front_pixels')
        actual=camera_position(base._model.cam_quat[target])
        np.testing.assert_allclose(actual,POSITION,atol=1e-12,rtol=0)
        np.testing.assert_allclose(base._model.cam_quat[target],QUATERNION,atol=1e-12,rtol=0)
        base._model.cam_pos[target]=actual
        return dict(result,source_position=SOURCE_POSITION,position=actual.tolist(),pullback=.1,crop=None)
    core.camera_for_group=camera_for_group;core.configure_camera=configure_camera
    core.quad_registration_proof=dict(source_sha256=SOURCE_SHA,strict_dictionary_edits=2,
        inverse_source_exact=True,original_policy_loader_latent192_and_evaluate_unchanged=True)
    return core
