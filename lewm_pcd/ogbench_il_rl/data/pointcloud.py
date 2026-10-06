"""Fixed crop foreground sampling, with authorized real-foreground empty fallback."""
import numpy as np

def sample_source_recipe(xyz, rgb, cube, uv, geom_id, seed):
    import numpy as np
    low, high = np.array([.10, -.50, -.02]), np.array([.75, .50, .60])
    inside = np.all((xyz >= low) & (xyz <= high), axis=1)
    xyz, rgb, cube, uv, geom_id = (value[inside] for value in (xyz, rgb, cube, uv, geom_id))
    cropped = len(xyz)
    _, keep = np.unique(np.floor(xyz / .005).astype(np.int32), axis=0, return_index=True)
    xyz, rgb, cube, uv, geom_id = (value[keep] for value in (xyz, rgb, cube, uv, geom_id))
    if not len(xyz):
        raise ValueError('Actual cropped/voxel pool empty; no fabricated zero cloud')
    rng = np.random.default_rng(seed)
    ci, ri = np.flatnonzero(cube), np.flatnonzero(~cube)
    n_cube = min(300, len(ci))
    if not len(ri):
        selected = rng.choice(ci, 1200, replace=len(ci) < 1200)
    else:
        selected = np.concatenate([rng.choice(ci, n_cube, replace=False), rng.choice(ri, 1200 - n_cube, replace=len(ri) < 1200 - n_cube)])
    selected = rng.permutation(selected)
    receipt = {'foreground_cropped_pool': cropped, 'voxel_unique_pool': len(xyz), 'cube_unique_pool': len(ci), 'other_unique_pool': len(ri), 'sampled_cube_points': int(cube[selected].sum()), 'actual_selected_unique_pool_indices': len(np.unique(selected)), 'repeated_samples': int(1200 - len(np.unique(selected))), 'replacement_fraction': float((1200 - len(np.unique(selected))) / 1200), 'sampling_seed': seed, 'quota_rule': 'original n_cube=min(round(.25*1200),cube_pool); never fake scarce cube points', 'crop_box': [low.tolist(), high.tolist()], 'voxel_size': .005}
    return xyz[selected].astype(np.float32), rgb[selected].astype(np.uint8), uv[selected], geom_id[selected], receipt


LOW=np.array([.10,-.50,-.02]);HIGH=np.array([.75,.50,.60]);VOXEL=.005

def sample_foreground(xyz,rgb,cube,uv,geom_id,seed):
    n=len(xyz)
    if not n:raise ValueError('Raw foreground empty; no background or fabricated cloud fallback')
    if xyz.shape!=(n,3) or rgb.shape!=(n,3) or rgb.dtype!=np.uint8 or cube.shape!=(n,) or uv.shape!=(n,2) or geom_id.shape!=(n,) or not np.isfinite(xyz).all():
        raise ValueError('Actual foreground shape/dtype/nonfinite contract failed')
    inside=np.all((xyz>=LOW)&(xyz<=HIGH),axis=1);cropped=int(inside.sum())
    if cropped:
        result=sample_source_recipe(xyz,rgb,cube,uv,geom_id,seed)
        receipt={**result[4],'fallback_used':False,'fallback_reason':None,'raw_foreground_pool':n,
                 'actual_sampling_domain':'original fixed crop foreground','foreground_cropped_pool':cropped}
        return *result[:4],receipt
    # Identical voxel/quota/seed/replacement machinery on the real uncropped
    # foreground. No points, colors, background, camera or source rows invented.
    _,keep=np.unique(np.floor(xyz/VOXEL).astype(np.int32),axis=0,return_index=True)
    xyz,rgb,cube,uv,geom_id=(value[keep] for value in (xyz,rgb,cube,uv,geom_id))
    if not len(xyz):raise ValueError('Actual uncropped foreground voxel pool empty')
    rng=np.random.default_rng(seed);ci,ri=np.flatnonzero(cube),np.flatnonzero(~cube);n_cube=min(300,len(ci))
    if not len(ri):selected=rng.choice(ci,1200,replace=len(ci)<1200)
    else:selected=np.concatenate([rng.choice(ci,n_cube,replace=False),rng.choice(ri,1200-n_cube,replace=len(ri)<1200-n_cube)])
    selected=rng.permutation(selected);unique=len(np.unique(selected))
    receipt={'foreground_cropped_pool':0,'raw_foreground_pool':n,'voxel_unique_pool':len(xyz),'cube_unique_pool':len(ci),'other_unique_pool':len(ri),
      'sampled_cube_points':int(cube[selected].sum()),'actual_selected_unique_pool_indices':unique,'repeated_samples':1200-unique,
      'replacement_fraction':(1200-unique)/1200,'sampling_seed':seed,'quota_rule':'original n_cube=min(round(.25*1200),cube_pool); never fake scarce cube points',
      'crop_box':[LOW.tolist(),HIGH.tolist()],'voxel_size':VOXEL,'fallback_used':True,
      'fallback_reason':'original_crop_empty_use_actual_same_camera_foreground','actual_sampling_domain':'uncropped actual foreground only',
      'input_domain_difference':'fallback xyz may lie outside self-collected pretrained fixed crop; frozen encoder numeric gate does not prove downstream SR'}
    return xyz[selected].astype(np.float32),rgb[selected].astype(np.uint8),uv[selected],geom_id[selected],receipt
