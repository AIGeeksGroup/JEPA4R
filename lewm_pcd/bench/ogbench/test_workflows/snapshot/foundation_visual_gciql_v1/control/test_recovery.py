"""CPU-only frozen-prefix adoption contract tests; no model imports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from barrier import validate_plan, _require_adoption, sha
from build_queues import make_plans


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.runtime=self.root/'old/runtime';self.runtime.mkdir(parents=True)
        self.source=self.root/'producer.py';self.source.write_text('original')
        self.old=make_plans({str(self.source):sha(self.source)},['visreg_vitb768','vjepa21_vitl1024'],'random_init_v1_','random_smoke')['smoke']
        for phase in self.old['phases']:
            if phase['name'].endswith('_02_deliver_pilot'):
                args=phase['argv'];i=args.index('--producer-exit-file')+1
                args[i]=str(self.runtime/Path(args[i]).name)
        self.path=self.root/'old/launch_plan.json';self.write(self.path,self.old)
        self.write(self.runtime/'actual_dispatch.json',dict(launch_plan_sha256=sha(self.path)))
        (self.runtime/'outer_exit_code').write_text('1\n')
        for phase in self.old['phases'][:6]:
            self.write(self.runtime/(phase['name']+'_exit.json'),dict(actual_exit_code=0,residual_owned_members=[]))
            self.write(self.runtime/(phase['name']+'_argv.json'),dict(argv=phase['argv']))
        self.plan=copy.deepcopy(self.old);self.plan['phases']=self.plan['phases'][6:]
        self.plan['recovery_runtime']=str(self.root/'new/runtime')
        for phase in self.plan['phases']:
            if phase['name'].endswith('_02_deliver_pilot'):
                args=phase['argv'];i=args.index('--producer-exit-file')+1
                args[i]=str(Path(self.plan['recovery_runtime'])/Path(args[i]).name)
        self.plan['adoption']=dict(plan_path=str(self.path),plan_sha256=sha(self.path),runtime=str(self.runtime),phase_count=6,
            evidence={str(p):sha(p) for p in self.runtime.iterdir()},changed_source_backups={})
        self.plan['adoption']['evidence'][str(self.path)]=sha(self.path)

    def write(self,path,value):path.write_text(json.dumps(value))

    def test_six_plus_fifty_eight(self):
        validate_plan(self.plan);self.assertEqual(_require_adoption(self.plan),6)

    def test_failed_prefix_exit_rejected(self):
        path=self.runtime/(self.old['phases'][0]['name']+'_exit.json')
        self.write(path,dict(actual_exit_code=1,residual_owned_members=[]))
        self.plan['adoption']['evidence'][str(path)]=sha(path)
        with self.assertRaises(AssertionError):_require_adoption(self.plan)

    def test_omitted_phase_rejected(self):
        self.plan['phases'].pop()
        with self.assertRaises(AssertionError):validate_plan(self.plan)

    def test_changed_source_requires_original_backup(self):
        backup=self.root/'producer.before.py';backup.write_bytes(self.source.read_bytes())
        self.source.write_text('changed')
        with self.assertRaises(AssertionError):_require_adoption(self.plan)
        self.plan['adoption']['changed_source_backups'][str(self.source)]=str(backup)
        self.assertEqual(_require_adoption(self.plan),6)

    def test_argv_change_rejected(self):
        self.plan['phases'][0]['argv'].append('--unapproved')
        with self.assertRaises(AssertionError):validate_plan(self.plan)


if __name__=='__main__':unittest.main()
