"""Pure CPU view-aware box gate; leaves image/camera/50um surface threshold unchanged."""
import numpy as np

SURFACE_TOLERANCE_M = 5e-5

def box_pixel_rays(position, rotation, size, camera_R, camera_t, focal, width, height):
    vv,uu=np.mgrid[:height,:width];cx,cy=(width-1)/2,(height-1)/2
    rays=np.stack([(uu-cx)/focal,(vv-cy)/focal,np.ones_like(uu)],-1)@camera_R.T@rotation
    camera=(camera_t-position)@rotation
    with np.errstate(divide='ignore',invalid='ignore'):
        spans=np.sort(np.stack([(-size-camera)/rays,(size-camera)/rays]),axis=0)
    entry,exit_=spans[0].max(-1),spans[1].min(-1)
    mask=(entry<=exit_)&(entry>0)&np.isfinite(entry)
    return mask,entry

def validate_view(*, position, rotation, size, camera_R, camera_t, focal, depth, geom_ids, cube_geom_id, visible_surface_max_error_m):
    if visible_surface_max_error_m is not None and (not np.isfinite(visible_surface_max_error_m) or visible_surface_max_error_m>SURFACE_TOLERANCE_M):
        raise ValueError('Visible cube surface exceeds unchanged 50um threshold')
    height,width=depth.shape;cx,cy=(width-1)/2,(height-1)/2
    c=(position-camera_t)@camera_R
    if not np.isfinite(c).all() or c[2]<=0:raise ValueError('Cube is behind camera; this branch is not defined')
    uv=np.array([focal*c[0]/c[2]+cx,focal*c[1]/c[2]+cy]);px,py=np.rint(uv).astype(int)
    inframe=bool(0<=px<width and 0<=py<height)
    predicted,entry=box_pixel_rays(position,rotation,size,camera_R,camera_t,focal,width,height)
    actual=geom_ids==cube_geom_id
    if np.any(actual & ~predicted):raise ValueError('Cube segmentation outside analytic integer-ray silhouette')
    if inframe:
        if geom_ids[py,px]==cube_geom_id:mode='in_frame_center_hit'
        elif predicted[py,px] and geom_ids[py,px]>=0 and np.isfinite(depth[py,px]) and entry[py,px]-depth[py,px]>SURFACE_TOLERANCE_M:
            mode='in_frame_center_proven_occlusion'
        else:raise ValueError('In-frame center gate still fails')
    else:
        # A center outside the image cannot have a nearest segmentation pixel.
        # Validate all in-image box rays instead of inventing/clamping that pixel.
        noncube=predicted & ~actual
        if np.any(noncube & ((geom_ids<0)|~np.isfinite(depth)|(entry-depth<=SURFACE_TOLERANCE_M))):
            raise ValueError('Out-of-frame center: a predicted box pixel lacks closer-geometry proof')
        if actual.any():mode='out_of_frame_center_partially_visible_box'
        elif predicted.any():mode='out_of_frame_center_remaining_box_pixels_proven_occluded'
        else:mode='out_of_frame_center_no_box_pixel_intersects_frame'
    gaps=entry[predicted & ~actual]-depth[predicted & ~actual]
    return {'branch':mode,'projected_center_uv':uv.tolist(),'center_in_frame':inframe,
            'analytic_box_pixels':int(predicted.sum()),'visible_cube_pixels':int(actual.sum()),
            'surface_tolerance_m':SURFACE_TOLERANCE_M,
            'noncube_box_ray_min_closer_depth_m':float(gaps.min()) if len(gaps) else None}
