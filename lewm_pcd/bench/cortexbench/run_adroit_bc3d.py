"""Thin Adroit PCD entry: original run_bc.py and official BC loop unchanged."""
import importlib.util
import os
from pathlib import Path
import runpy
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def task_from_env_id(env_id):
    # Match a whole suffix component, including the registered collection IDs.
    suffix = env_id.rsplit(':', 1)[-1].rsplit('/', 1)[-1]
    tasks = {'pen-v0': 'pen-v0', 'relocate-v0': 'relocate-v0',
             'AdroitPen-v0': 'pen-v0', 'AdroitRelocate-v0': 'relocate-v0'}
    if suffix not in tasks:
        raise ValueError(f'unsupported Adroit environment ID: {env_id}')
    return tasks[suffix]


def load_backend():
    path = HERE.parents[1] / 'envs/adroit/pcd.py'
    spec = importlib.util.spec_from_file_location('adroit_pcd_backend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_image(self):
    assert self.width == self.height == 256, 'Adroit PCD BC requires the original 256x256 camera'
    pcm = getattr(self, '_pcm', None)
    if pcm is None:
        pcm = self._pcm = load_backend().AdroitPointClouds(self.sim, task_from_env_id(self.spec.id), size=256)
        if getattr(self, '_pcd_seed', None) is not None:
            pcm.seed(self._pcd_seed)
    xyz, rgb = pcm()
    return np.concatenate([xyz, rgb.astype(np.float32)], axis=1)


def seed(self, seed=None):
    self._pcd_seed = seed
    if getattr(self, '_pcm', None) is not None:
        self._pcm.seed(seed)
    return self.env.seed(seed)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    env_args = [arg for arg in argv if arg.startswith('env=')]
    if len(env_args) != 1:
        raise ValueError('provide exactly one env=... override')
    argv[argv.index(env_args[0])] = 'env=' + task_from_env_id(env_args[0].split('=', 1)[1])
    if '--config-name' not in argv and not any(arg.startswith('--config-name=') for arg in argv):
        argv = ['--config-name', 'Adroit_BC_config'] + argv
    selected = next((arg.split('=',1)[1] for arg in argv if arg.startswith('--config-name=')), None)
    if '--config-name' in argv:
        selected = argv[argv.index('--config-name')+1]
    if selected != 'Adroit_BC_config':
        raise ValueError('Adroit entry requires the existing Adroit_BC_config')
    if len([arg for arg in argv if arg.startswith('hydra.run.dir=')]) != 1:
        raise ValueError('provide exactly one explicit fresh hydra.run.dir=...')
    for arg in argv:
        if arg.startswith('hydra.job.chdir=') and arg.split('=',1)[1].lower() != 'true':
            raise ValueError('Adroit BC requires the official chdir into its run directory')
        if arg.startswith('job_name='):
            name=arg.split('=',1)[1]
            if name.startswith(('/', '\\')) or Path(name).is_absolute() or '..' in name.replace('\\','/').split('/'):
                raise ValueError('job_name must stay inside the fresh run directory')
        if arg.startswith('hydra.run.dir='):
            directory = Path(arg.split('=',1)[1]).expanduser()
            if directory.exists() and any(directory.iterdir()):
                raise FileExistsError(f'existing BC output: {directory}')
    vi = HERE.parents[1] / 'third_party/eai-vc/cortexbench/mujoco_vc/visual_imitation'
    sys.path.insert(0, str(vi)); sys.path.insert(0, str(HERE))
    import hydra
    import mujoco_vc.gym_wrapper as gw
    import mujoco_vc.model_loading as ml
    from omegaconf import OmegaConf
    import train_loop

    def load_pretrained_model(embedding_name, input_type=np.ndarray, *args, **kwargs):
        config = OmegaConf.load(os.path.join(ml.vc_models_dir_path, 'conf/model', embedding_name + '.yaml'))
        model, embedding_dim, transforms, metadata = hydra.utils.call(config)
        if metadata.get('input') != 'pcd':
            raise ValueError('PCD entry requires a PCD embedding; RGB uses the original run_bc.py')
        return model.eval(), embedding_dim, (lambda pcd: transforms(pcd).unsqueeze(0)), metadata

    # Same input-transform bridge as the original PCD entry; no learning changes.
    for module in (ml, gw, train_loop):
        module.load_pretrained_model = load_pretrained_model
    gw.MuJoCoPixelObsWrapper.get_image = get_image
    gw.MuJoCoPixelObsWrapper.seed = seed
    sys.argv = [str(HERE / 'run_bc.py')] + argv
    runpy.run_path(str(HERE / 'run_bc.py'), run_name='__main__')


if __name__ == '__main__':
    main()
