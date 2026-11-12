"""Zero-update checks of the real retained input and derived VDS mappings."""
import copy,importlib.util,json,os,sys,tempfile,unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from bench.ogbench import retained_training as retained


@unittest.skipUnless(importlib.util.find_spec('h5py')and os.environ.get('CUBE_RETAINED_MANIFEST'),
                     'requires the explicitly supplied real retained manifest and existing h5py')
class RetainedHDFTests(unittest.TestCase):
    def setUp(self):
        import hdf5plugin
        self.path=Path(os.environ['CUBE_RETAINED_MANIFEST'])
        self.value=retained.read(self.path)
        self.desc=retained.read(retained.pinned(self.value['retained_import']))

    def changed_descriptor(self,folder,change):
        desc=copy.deepcopy(self.desc);change(desc)
        path=Path(folder)/'changed_import.json';retained.write(path,desc)
        value=copy.deepcopy(self.value);value['retained_import']=retained.pin(path)
        return value

    def test_complete_real_input_validates_and_all_228_original_files_are_mapped(self):
        from bench.ogbench import fresh_data
        fresh_data.load_binding(self.path)
        self.assertEqual(len(self.desc['producer_aliases']),228)
        self.assertEqual(self.desc['unavailable_non_executable_metadata'],[])
        self.assertEqual(len(self.desc['raw_shards']),6)

    def test_missing_original_python_is_rejected(self):
        with tempfile.TemporaryDirectory()as folder:
            def remove(desc):
                name=next(n for n in desc['producer_aliases']if n.endswith('.py'))
                del desc['producer_aliases'][name]
            value=self.changed_descriptor(folder,remove)
            with self.assertRaisesRegex(ValueError,'unmapped original producer'):retained.validate_manifest(value,self.path)

    def test_raw_part_or_modality_rebinding_is_rejected(self):
        with tempfile.TemporaryDirectory()as folder:
            value=self.changed_descriptor(folder,lambda d:d['raw_shards'].reverse())
            with self.assertRaisesRegex(ValueError,'order/modality'):retained.validate_manifest(value,self.path)

    def test_swapped_actual_train_shards_are_rejected(self):
        from bench.ogbench.collection import merge_raw
        with tempfile.TemporaryDirectory()as folder:
            folder=Path(folder)
            raw=[r['actual']['path']for r in self.desc['raw_shards']if r['index']in(0,1)and r['modality']=='rgb']
            wrong=merge_raw(raw[::-1],folder/'reversed_train_rgb.h5')
            value=self.changed_descriptor(folder,lambda d:d['derived_raw'].update(train_rgb=wrong))
            value['raw']['train_rgb']=wrong
            with self.assertRaisesRegex(ValueError,'row/split order'):retained.validate_manifest(value,self.path)

    def test_including_one_terminal_row_in_training_is_rejected(self):
        import h5py
        with tempfile.TemporaryDirectory()as folder:
            wrong=Path(folder)/'wrong_terminal.h5';key='train_rgb';raw=self.value['raw'][key]
            with h5py.File(raw,'r')as source,h5py.File(wrong,'x',libver='latest')as out:
                for name in self.value['views'][key]['keys']:
                    col=source[name];layout=h5py.VirtualLayout((400,*col.shape[1:]),col.dtype)
                    src=h5py.VirtualSource(raw,name,shape=col.shape)
                    layout[:201]=src[:201];layout[201:400]=src[201:400]
                    out.create_virtual_dataset(name,layout)
            value=copy.deepcopy(self.value);value['views'][key]['path']=str(wrong)
            with self.assertRaisesRegex(ValueError,'terminal-row mapping'):retained.validate_manifest(value,self.path)


if __name__=='__main__':unittest.main()
