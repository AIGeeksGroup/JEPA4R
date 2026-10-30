"""Counted generalization of the frozen original static renderer, N=1..4."""
from pathlib import Path
import types
from common import replace, sha

BASE=Path(__file__).parent/'base/render_adapter.py'
BASE_SHA='5df6dc694f7c9d38ccff69385080b1bad814f41ad6c4ae71e9def2e6ac71662f'

def source():
    assert sha(BASE)==BASE_SHA
    text=BASE.read_text()
    text=replace(text,"    import atexit\n","    import atexit\n    STATE['cube_count']=camera['cube_count']\n    n=camera['cube_count']\n")
    text=replace(text,"gym.make('cube-triple-v0'", "gym.make(camera['env_id']")
    text=replace(text,'(35, 32, 7)','(14+7*n, 14+6*n, 7)')
    text=replace(text,"mujoco.mjtObj.mjOBJ_CAMERA, 'front')","mujoco.mjtObj.mjOBJ_CAMERA, camera['source_name'])")
    text=replace(text,"camera['position'], atol=1e-12", "camera['source_position'], atol=1e-12")
    text=replace(text,'np.deg2rad(5.)',"np.deg2rad(camera['local_x_up_degrees'])")
    text=replace(text,'= position, actual, 45.',"= camera['position'], actual, 45.")
    text=replace(text,'range(3)',"range(STATE['cube_count'])",2)
    text=replace(text,"source_name='front'","source_name=camera['source_name']")
    text=replace(text,'pullback=0, crop=None',"pullback=camera['pullback'], crop=None")
    text=replace(text,'nq=35, nv=32, nu=7',"nq=14+7*STATE['cube_count'], nv=14+6*STATE['cube_count'], nu=7")
    text=replace(text,'(target + 1) % 3',"(target + 1) % STATE['cube_count']")
    return text

_m=types.ModuleType('_foundation_static_renderer');_m.__file__=str(BASE)
exec(compile(source(),str(BASE),'exec'),_m.__dict__)
# Spawn workers import the same thin bridge and reconstruct the same source.
STATE=_m.STATE
init_worker=_m.init_worker
render_rows=_m.render_rows
render_episode=_m.render_episode
missing_state_witness=_m.missing_state_witness
# Register the executable namespace so spawn can pickle its public functions.
import sys
sys.modules[_m.__name__]=_m
for _f in (init_worker,render_rows,render_episode,missing_state_witness):
    _f.__module__=__name__
