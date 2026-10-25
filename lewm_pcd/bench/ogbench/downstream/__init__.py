"""Task and encoder binding for the preserved Cube GCIQL/Native code."""
import copy
import math
import re
import sys

METHODS = {'state': ('state', 0), 'pldm': ('pldm', 512),
           'lejepa': ('lejepa', 512), 'lewm_rgb': ('lewm', 192), 'lewm_pcd': ('lewm', 192)}
PROTOCOL = 'cube_all_distance_v1'


def bind_recipe(profile, method, config):
    """Validate actual widths and protocol; never infer a checkpoint or camera."""
    if method not in METHODS:
        raise ValueError(f'Unsupported downstream method: {method}')
    cubes = profile['cube_count']
    if cubes not in (1, 2, 3, 4) or config.get('cube_count') != cubes:
        raise ValueError('Recipe and profile must identify the same Cube1..4 task')
    if config.get('protocol') != PROTOCOL:
        raise ValueError('Only cube_all_distance_v1 is supported here; use the preserved entry for official state GCIQL')
    family, dim = METHODS[method]
    modality = 'state' if not dim else 'pcd' if method == 'lewm_pcd' else 'rgb'
    if config.get('modality') != modality:
        raise ValueError('Recipe modality does not match the selected method')
    expected = dict(current_dim=19+9*cubes if not dim else dim+19,
                    goal_dim=3*cubes if not dim else dim+3*cubes,
                    latent_dim=dim, zero_padding=False)
    if config.get('input_contract') != expected:
        raise ValueError(f'Actual dimensions required without synthetic padding: {expected}')
    encoder = config.get('encoder_binding', {})
    if encoder.get('family') != family or encoder.get('latent_dim') != dim:
        raise ValueError('Encoder family and output width do not match the recipe')
    digest = encoder.get('checkpoint_sha256')
    if dim and (not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None):
        raise ValueError('An exact encoder checkpoint SHA256 is required')
    if not dim and digest is not None:
        raise ValueError('State-only uses no encoder checkpoint')
    if dim:
        preprocessing = encoder.get('preprocessing', {})
        if (preprocessing.get('input_dtype') != ('float32' if modality == 'pcd' else 'uint8') or
                preprocessing.get('layout') != ('XYZRGB1200' if modality == 'pcd' else 'HWC224') or
                preprocessing.get('device') not in ('cpu', 'cuda') or
                re.fullmatch('[0-9a-f]{64}', str(preprocessing.get('source_sha256', ''))) is None):
            raise ValueError('Explicit accepted preprocessing source, dtype, layout, and device required')
        if family in ('lejepa', 'lewm') and preprocessing['device'] != 'cpu':
            raise ValueError('Preserved Tiny/LeWM Native preprocessing executes on CPU')
        if modality == 'pcd' and (preprocessing.get('center') != [.425, 0., .2] or
                preprocessing.get('scale') != 4. or preprocessing.get('use_rgb') is not True):
            raise ValueError('Preserved PCD normalization requires explicit center [.425,0,.2], scale4 and RGB')
    native = config.get('native_protocol', {})
    horizon = native.get('max_episode_steps')
    if native.get('evaluator') != 'ogbench_native_v1' or type(horizon) is not int or horizon < 1:
        raise ValueError('Explicit OGBench Native evaluator and positive episode horizon required')
    if profile.get('max_episode_steps', horizon) != horizon:
        raise ValueError('Native horizon differs from the selected profile')
    if 'camera' not in native or (dim and not isinstance(native['camera'], dict)):
        raise ValueError('Explicit camera binding required for visual Native inputs')
    if dim:
        camera = native['camera']
        profile_camera = profile.get('camera', {})
        if (camera.get('width') != 224 or camera.get('height') != 224 or camera.get('aa_samples') != 4 or
                type(camera.get('fov')) not in (int, float) or not math.isfinite(camera['fov']) or not 0 < camera['fov'] < 180 or
                type(camera.get('pullback_m')) not in (int, float) or camera['pullback_m'] not in (0., .1) or
                not isinstance(camera.get('position'), list) or len(camera['position']) != 3 or
                not all(type(x) in (int, float) and math.isfinite(x) for x in camera['position'])):
            raise ValueError('Explicit finite camera pose/FOV and preserved224/AA4 capture required')
        if cubes == 1:
            if (profile_camera.get('source_name') != 'front_pixels' or profile_camera.get('render_name') != 'front_pixels' or
                    camera.get('source_name') != 'front_pixels' or camera.get('render_name') != 'front_pixels' or
                    camera.get('local_x_up_degrees') != 0. or camera['pullback_m'] != 0.):
                raise ValueError('Single requires its declared front_pixels identity camera, no extra rotation or pullback')
        elif (camera.get('source_name') != 'front' or camera.get('render_name') != 'front_pixels' or
                camera['fov'] != 45. or camera.get('local_x_up_degrees') != 5.):
            raise ValueError('Cube2..4 require the preserved front/FOV45/local-X5 camera')
        for key in ('source_name', 'render_name', 'local_x_up_degrees', 'pullback_m', 'fov'):
            if key in profile_camera and camera.get(key) != profile_camera[key]:
                raise ValueError('Camera differs from declared profile: '+key)
        if profile_camera.get('pose_runtime_verified') is not False and 'position' in profile_camera:
            if camera['position'] != profile_camera['position']:
                raise ValueError('Camera position differs from the declared profile')
        if camera['pullback_m'] and camera.get('source_position') != [1.287, 0., .509]:
            raise ValueError('Pullback requires the explicit original front source position')
        if modality == 'pcd':
            crop = camera.get('pointcloud_crop_box')
            if (not isinstance(crop, list) or len(crop) != 2 or
                    any(not isinstance(row, list) or len(row) != 3 for row in crop) or
                    any(type(x) not in (int, float) or not math.isfinite(x) for row in crop for x in row) or
                    any(lo >= hi for lo, hi in zip(crop[0], crop[1]))):
                raise ValueError('Explicit finite point-cloud crop bounds required')
    if native.get('encoding_protocol') != 'fresh_current_and_goal_each_policy_call_2N':
        raise ValueError('Native requires fresh current and goal encoding for every policy call')
    result = copy.deepcopy(config)
    result.update(method=method, cube_count=cubes, state_dim=19+9*cubes,
                  latent_dim=dim, encoder_family=family, modality=modality,
                  source_profile=profile.get('profile_name', profile.get('name')),
                  source=profile.get('source'))
    return result


def build_command(profile_name, method, stage, data_root, output_root, config_path, python=None):
    from ..contracts import get_profile
    get_profile(profile_name)
    if method not in METHODS or stage not in ('gciql', 'native-sr'):
        raise ValueError('Supported downstream stages are gciql and native-sr')
    if method == 'state' and stage == 'native-sr':
        raise ValueError('The shared preserved visual Native loop requires a visual method; use the original state Native entry')
    if config_path is None:
        raise ValueError('A protocol-bound downstream recipe is required')
    return [str(python or sys.executable), '-m', 'bench.ogbench.downstream.runtime',
            '--profile', str(profile_name), '--method', method, '--stage', stage,
            '--data-root', str(data_root), '--output-root', str(output_root), '--config', str(config_path)]


def validate_cache_protocol(manifest, config):
    """Tie online camera/preprocessing to the SHA-pinned offline producer protocol."""
    protocol = manifest.get('protocol', {})
    if (protocol.get('encoder_family', 'lewm'), protocol.get('latent_dim')) != (
            config['encoder_family'], config['latent_dim']):
        raise ValueError('Feature cache family/output width mismatch')
    camera = protocol.get('camera', {})
    expected = config['native_protocol']['camera']
    for key in ('source_name', 'render_name', 'position', 'fov', 'local_x_up_degrees', 'width', 'height', 'aa_samples'):
        if camera.get(key) != expected[key]:
            raise ValueError('Offline and online camera differ: '+key)
    if camera.get('pullback_m', camera.get('pullback', 0.)) != expected['pullback_m']:
        raise ValueError('Offline and online camera pullback differ')
    preprocessing = config['encoder_binding']['preprocessing']
    if protocol.get('pixel_preprocess_device') != preprocessing['device']:
        raise ValueError('Offline and online preprocessing devices differ')
    if preprocessing['source_sha256'] not in protocol.get('source_pins', {}).values():
        raise ValueError('Preprocessing code is absent from the actual cache producer source pins')
    if config['modality'] == 'pcd':
        if camera.get('pointcloud_crop_box', camera.get('crop_box')) != expected['pointcloud_crop_box']:
            raise ValueError('Offline and online PCD crop differ')
