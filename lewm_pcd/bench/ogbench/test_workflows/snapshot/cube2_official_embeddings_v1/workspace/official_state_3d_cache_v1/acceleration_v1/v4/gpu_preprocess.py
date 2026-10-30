"""Original XYZ-only FP32 __call__ plus exact CPU-domain uint8 RGB lookup on GPU."""


def install(encoder):
    import torch
    encoder.source_paths()
    from utils import get_pcd_preprocessor
    cpu_operation = get_pcd_preprocessor(center=(.425, 0., .2), scale=4., use_rgb=True)
    colors = torch.arange(256, dtype=torch.uint8)[:, None, None].expand(-1, 1, 3)
    lookup = cpu_operation(dict(pcd_xyz=torch.zeros((256, 1, 3), dtype=torch.float32),
                                pcd_rgb=colors))[cpu_operation.target][:, 0, 3].to('cuda')
    operation = get_pcd_preprocessor(center=(.425, 0., .2), scale=4., use_rgb=False)
    operation.center = operation.center.to('cuda')
    original = encoder.preprocess

    def preprocess(batch, modality):
        if modality != '3d':
            return original(batch, modality)
        assert batch['pcd_xyz'].dtype == torch.float32 and batch['pcd_rgb'].dtype == torch.uint8
        assert batch['pcd_xyz'].shape == batch['pcd_rgb'].shape
        rgb = batch['pcd_rgb'].to('cuda')
        result = operation(dict(pcd_xyz=batch['pcd_xyz'].to('cuda'), pcd_rgb=rgb))
        result[operation.target] = torch.cat((result[operation.target], lookup[rgb.long()]), dim=-1)
        return result

    encoder.preprocess = preprocess
    return original
