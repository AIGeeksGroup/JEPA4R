"""One authorized real Cube1 AA4 frame; no encoder/model imports or execution."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/root/autodl-tmp/workspace/users/lewm/cube_downstream_20261017_v1')
CASE=ROOT/'foundation_visual_gciql_v1'
OUT=Path(__file__).resolve().parent

def write(name,value):
    with (OUT/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)

def worker():
    import numpy as np
    sys.path.insert(0,str(CASE/'cache'))
    import render_adapter as render
    camera=json.loads((CASE/'cache/data/cube1/camera.json').read_text())
    arrays=CASE/'cache/data/cube1/source_arrays.json'
    env=renderer=None
    report=dict(started=time.time(),frames_requested=1,encoder_calls=0,training_updates=0)
    try:
        render.init_worker(str(ROOT),str(arrays),camera)
        env=render.STATE['env'];base=env.unwrapped
        assert base._renderer is None,'No prior lazy renderer expected'
        result=render.render_rows(('train',np.array([0],np.int64)))
        renderer=base._renderer;pixels=result['pixels']
        assert renderer is not None
        assert pixels.shape==(1,224,224,3) and pixels.dtype==np.uint8
        assert np.isfinite(pixels).all() and int(pixels.max())>int(pixels.min())
        assert float(pixels.std())>0 and int(base._model.vis.quality.offsamples)==4
        report.update(actual_exit_code=0,frames_rendered=1,shape=list(pixels.shape),dtype=str(pixels.dtype),
            finite=True,pixel_min=int(pixels.min()),pixel_max=int(pixels.max()),pixel_std=float(pixels.std()),
            unique_RGB_colors=int(len(np.unique(pixels.reshape(-1,3),axis=0))),
            RGB_sha256=hashlib.sha256(memoryview(pixels)).hexdigest(),camera=result['camera'],
            geometry=result['render_geometry'],no_model_forward=True,
            MUJOCO_GL=os.environ.get('MUJOCO_GL'),PYOPENGL_PLATFORM=os.environ.get('PYOPENGL_PLATFORM'),
            LD_LIBRARY_PATH=os.environ.get('LD_LIBRARY_PATH'))
    finally:
        if renderer is not None:
            renderer.close()
            assert renderer._gl_context is None and renderer._mjr_context is None
            report['renderer_GL_and_MJR_contexts_freed']=True
        if env is not None:
            env.close();report['env_close_returned']=True
        report['completed']=time.time()
        write('render_report.json',report)
    print(json.dumps(report),flush=True)

def main():
    env=dict(os.environ,LD_LIBRARY_PATH=str(ROOT/'runtime/lib')+':/usr/lib/x86_64-linux-gnu',
             MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker']
    write('argv.json',dict(argv=argv,cwd=str(ROOT),environment={k:env[k] for k in
        ('LD_LIBRARY_PATH','MUJOCO_GL','PYOPENGL_PLATFORM','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    started=time.time()
    with (OUT/'stdout.log').open('x') as stdout,(OUT/'stderr.log').open('x') as stderr:
        child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        write('child_identity.json',dict(pid=child.pid,pgid=child.pid,started=started))
        code=child.wait(timeout=120)
    rows=subprocess.run(['ps','-eo','pid=,pgid=,args='],capture_output=True,text=True,check=True).stdout
    residual=[]
    for row in rows.splitlines():
        fields=row.split(None,2)
        if len(fields)==3 and int(fields[1])==child.pid:residual.append(dict(pid=int(fields[0]),pgid=int(fields[1]),args=fields[2]))
    write('actual_exit.json',dict(actual_exit_code=code,residual_owned_members=residual,started=started,
        completed=time.time(),child_pid=child.pid,pgid=child.pid))
    assert code==0 and residual==[],(code,residual)
    print(json.dumps(dict(actual_exit_code=code,residual_owned_members=residual,report=str(OUT/'render_report.json'))))

if __name__=='__main__':
    worker() if '--worker' in sys.argv else main()
