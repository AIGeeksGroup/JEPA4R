"""Thin Adroit PCD entry: original run_bc.py and official BC loop unchanged."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def task_from_env_id(env_id):
    # Match a whole suffix component, including the registered collection IDs.
    suffix = env_id.rsplit(':', 1)[-1].rsplit('/', 1)[-1]
    tasks = {'relocate-v0': 'relocate-v0', 'AdroitRelocate-v0': 'relocate-v0'}
    if suffix not in tasks:
        raise ValueError(f'unsupported Adroit environment ID: {env_id}')
    return tasks[suffix]


def load_backend():
    path = HERE.parents[1] / 'envs/adroit/pcd.py'
    spec = importlib.util.spec_from_file_location('adroit_pcd_backend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_point_clouds(sim, task):
    """One explicit sampling contract for official replay and online BC."""
    task = task_from_env_id(task)
    path = os.environ.get('ADROIT_PCD_CONFIG')
    kwargs = {'size': 256}
    if path:
        payload = Path(path).read_bytes()
        expected = os.environ.get('ADROIT_PCD_CONFIG_SHA256')
        if expected and hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Adroit PCD configuration SHA mismatch')
        config = json.loads(payload)
        kwargs.update(config.get('pcd_constructor_kwargs', config))
        if task_from_env_id(kwargs.pop('task', task)) != task:
            raise ValueError('Adroit PCD configuration task mismatch')
    return load_backend().AdroitPointClouds(sim, task, **kwargs)


def get_image(self):
    assert self.width == self.height == 256, 'Adroit PCD BC requires the original 256x256 camera'
    pcm = getattr(self, '_pcm', None)
    if pcm is None:
        pcm = self._pcm = build_point_clouds(self.sim, self.spec.id)
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
    eai = Path(os.environ.get('CORTEXBENCH_EAI_VC_ROOT', HERE.parents[1] / 'third_party/eai-vc'))
    vi = eai / 'cortexbench/mujoco_vc/visual_imitation'
    sys.path.insert(0, str(vi)); sys.path.insert(0, str(HERE))
    import hydra
    import mujoco_vc.gym_wrapper as gw
    import mujoco_vc.model_loading as ml
    from omegaconf import OmegaConf
    import train_loop

    def load_pretrained_model(embedding_name, input_type=np.ndarray, *args, **kwargs):
        config = OmegaConf.load(os.path.join(ml.vc_models_dir_path, 'conf/model', embedding_name + '.yaml'))
        checkpoint = os.environ.get('LEWM_PCD_CHECKPOINT_DIR')
        if checkpoint:
            if config.get('_target_') != 'lewm_cortex.load_lewm_pcd':
                raise ValueError('checkpoint override requires the LeWM PCD embedding')
            config.checkpoint_dir = checkpoint
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
