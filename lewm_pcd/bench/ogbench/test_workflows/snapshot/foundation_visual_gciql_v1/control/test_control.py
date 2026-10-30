import copy
import json
from pathlib import Path
import tempfile
import unittest
from barrier import validate_plan,require_smoke,sha
from build_queues import make_plans

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

class ControlTests(unittest.TestCase):
    def test_four_and_eight_group_paths_and_protocol(self):
        for families in (['vjepa21_vitl1024'],['visreg_vitb768','vjepa21_vitl1024']):
            plans=make_plans({},families)
            for kind,plan in plans.items():
                validate_plan(plan)
                self.assertEqual(len(plan['phases']),32*len(families))
                for phase in plan['phases']:
                    argv=phase['argv']
                    self.assertNotIn('TEMPLATE',' '.join(argv))
                    self.assertEqual(phase['environment']['PYOPENGL_PLATFORM'],'egl')
                    self.assertTrue(phase['environment']['LD_LIBRARY_PATH'].replace('\\','/').endswith('/runtime/lib:/usr/lib/x86_64-linux-gnu'))
                    self.assertEqual(phase['environment']['CUDA_VISIBLE_DEVICES'],'0' if 'gpu_lease.py' in ' '.join(argv) else '')
                    if '--producer-exit-file' in argv:
                        expected=phase['name'].replace('_02_deliver_pilot','_01_encode_pilot').replace('_03_deliver','_01_encode_full')
                        self.assertTrue(argv[argv.index('--producer-exit-file')+1].endswith(expected+'_exit.json'))
            bad=copy.deepcopy(plans['formal']);bad['training_updates']+=1
            with self.assertRaises(AssertionError):validate_plan(bad)

    def fixture(self,root):
        pin=root/'fixed_source.py';pin.write_text('original=1\n')
        plans=make_plans({str(pin):sha(pin)},['vjepa21_vitl1024'])
        smoke=plans['smoke'];directory=root/'smoke';runtime=directory/'runtime';runtime.mkdir(parents=True)
        for phase in smoke['phases']:
            if phase['name'].endswith('_08_native_audit'):
                config=root/'config'/f"{phase['name']}.json";write(config,{'fixture':True})
                output=root/'results'/phase['name']
                phase['argv'][phase['argv'].index('--config')+1]=str(config)
                phase['argv'][phase['argv'].index('--output')+1]=str(output)
                write(output/'independent_audit.json',dict(actual_exit_code=0,mode='tiny',diagnostic_excluded_from_formal_SR=True,config_sha256=sha(config)))
            write(runtime/(phase['name']+'_exit.json'),dict(actual_exit_code=0,residual_owned_members=[]))
            write(runtime/(phase['name']+'_argv.json'),dict(argv=phase['argv']))
        write(directory/'launch_plan.json',smoke)
        write(runtime/'closure.json',dict(actual_exit_code=0,completed_phases=len(smoke['phases']),families=smoke['families']))
        write(runtime/'actual_dispatch.json',dict(launch_plan_sha256=sha(directory/'launch_plan.json')))
        (runtime/'outer_exit_code').write_text('0\n')
        return directory,plans,pin

    def test_actual_completion_barrier_rejects_failure_residual_and_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory,plans,pin=self.fixture(Path(tmp))
            self.assertEqual(require_smoke(directory,plans['formal'])['groups'],4)
            phase=plans['smoke']['phases'][0];exitfile=directory/'runtime'/(phase['name']+'_exit.json')
            for record in (dict(actual_exit_code=1,residual_owned_members=[]),dict(actual_exit_code=0,residual_owned_members=[{'PID':1}])):
                write(exitfile,record)
                with self.assertRaises(AssertionError):require_smoke(directory,plans['formal'])
            write(exitfile,dict(actual_exit_code=0,residual_owned_members=[]));pin.write_text('changed=1\n')
            with self.assertRaises(AssertionError):require_smoke(directory,plans['formal'])

    def test_random_queue_has_no_formal_and_cannot_substitute_official(self):
        plans=make_plans({},['vjepa21_vitl1024'],'random_init_v1_','random_smoke')
        self.assertEqual(set(plans),{'smoke'})
        validate_plan(plans['smoke'])
        self.assertTrue(all(x['name'].startswith('random_init_v1_') for x in plans['smoke']['phases']))
        wrong=copy.deepcopy(plans['smoke']);wrong['queue']='formal'
        with self.assertRaises(AssertionError):validate_plan(wrong)

if __name__=='__main__':unittest.main()
