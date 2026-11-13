# Moved from OGBench-cube@4fbb0c6:lewm_pcd/bench/ogbench/downstream/__init__.py (verbatim)
"""Current Cube cache/online protocol binding."""
def validate_cache_protocol(manifest, config):
    """Tie online camera/preprocessing to the SHA-pinned offline producer protocol."""
    protocol = manifest.get('protocol', {})
    if (protocol.get('encoder_family', 'lewm'), protocol.get('latent_dim')) != (config['encoder_family'], config['latent_dim']):
        raise ValueError('Feature cache family/output width mismatch')
    camera = protocol.get('camera', {})
    expected = config['native_protocol']['camera']
    for key in ('source_name', 'render_name', 'position', 'fov', 'local_x_up_degrees', 'width', 'height', 'aa_samples'):
        if camera.get(key) != expected[key]:
            raise ValueError('Offline and online camera differ: ' + key)
    if camera.get('pullback_m', camera.get('pullback', 0.0)) != expected['pullback_m']:
        raise ValueError('Offline and online camera pullback differ')
    preprocessing = config['encoder_binding']['preprocessing']
    if protocol.get('pixel_preprocess_device') != preprocessing['device']:
        raise ValueError('Offline and online preprocessing devices differ')
    if preprocessing['source_sha256'] not in protocol.get('source_pins', {}).values():
        raise ValueError('Preprocessing code is absent from the actual cache producer source pins')
    if config['modality'] == 'pcd':
        if camera.get('pointcloud_crop_box', camera.get('crop_box')) != expected['pointcloud_crop_box']:
            raise ValueError('Offline and online PCD crop differ')
