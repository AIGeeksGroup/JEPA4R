"""PCD metadata adapter for the pinned shared frozen-latent reader."""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RGB = HERE.parent / 'rgb_latent_cache_20261012_v1'
if hashlib.sha256((RGB / 'cache_dataset.py').read_bytes()).hexdigest() != '0a4a73683ff1681eb11156187f096b11403882720db7de2655eaded3c8c5cae1':
    raise ValueError('Pinned shared latent reader changed')
sys.path.insert(0, str(RGB))
import cache_dataset as shared

def pcd_metadata_alias(value):
    if isinstance(value, list): return [pcd_metadata_alias(v) for v in value]
    if not isinstance(value, dict): return value
    result = {key: pcd_metadata_alias(v) for key, v in value.items()}
    for key in list(result):
        if key.startswith('source_pcd'): result['source_rgb' + key[len('source_pcd'):]] = result[key]
    return result

def read_pcd_metadata(path): return pcd_metadata_alias(json.loads(Path(path).read_text()))

shared.read = read_pcd_metadata

class CachedCubePCDLatentDataset(shared.CachedCubeStateLatentDataset):
    def __init__(self, cache_root, split, numeric_root, original_index, cache_seal):
        seal = json.loads(Path(cache_seal).read_text())
        manifest = json.loads(Path(seal['manifest_path']).read_text())
        if manifest.get('source_modality') != 'pcd' or seal.get('source_modality') != 'pcd':
            raise ValueError('Explicit sealed PCD modality required')
        super().__init__(cache_root, split, numeric_root, original_index, cache_seal)
