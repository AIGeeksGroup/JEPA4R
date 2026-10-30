import subprocess,json,datetime,pathlib,sys,os
p=json.loads(pathlib.Path(sys.argv[1]).read_text());o=pathlib.Path(sys.argv[1]).parent
f=(o/'run.log').open('x')
c=subprocess.Popen(p['argv'],cwd=p['cwd'],env=dict(os.environ,**p['environment']),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
(o/'child_pid').write_text(str(c.pid))
(o/'status.json').write_text(json.dumps(dict(stage='running',child_pid=c.pid,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=None)))
r=c.wait();f.close()
(o/'child_exit_code').write_text(str(r)+'\n');(o/'driver_exit_code').write_text(str(r)+'\n')
(o/'status.json').write_text(json.dumps(dict(stage='complete' if r==0 else 'failed',exit_code=r,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())))
sys.exit(r)
