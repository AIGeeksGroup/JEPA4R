"""Move XYZ FP32 and RGB uint8 separately; reuse the original preprocessor math."""


def install(encoder):
    import torch
    encoder.source_paths()
    from utils import get_pcd_preprocessor
    # Reuse exact CPU /255 results for the finite uint8 domain, avoiding CUDA's
    # one-ULP division rounding difference without sending FP32 RGB batches.
    cpu_operation = get_pcd_preprocessor(center=(.425, 0., .2), scale=4., use_rgb=True)
    colors = torch.arange(256, dtype=torch.uint8)[:, None, None].expand(-1, 1, 3)
    lookup = cpu_operation(dict(pcd_xyz=torch.zeros((256, 1, 3), dtype=torch.float32),
                                pcd_rgb=colors))[cpu_operation.target][:, 0, 3].to('cuda')
    operation = get_pcd_preprocessor(center=(.425, 0., .2), scale=4., use_rgb=True)
    operation.center = operation.center.to('cuda')
    original = encoder.preprocess

    def preprocess(batch, modality):
        if modality != '3d':
            return original(batch, modality)
        assert batch['pcd_xyz'].dtype == torch.float32 and batch['pcd_rgb'].dtype == torch.uint8
        assert batch['pcd_xyz'].shape == batch['pcd_rgb'].shape
        # XYZ remains the original FP32 __call__; only normalized RGB is replaced
        # with the exact CPU-domain lookup on GPU after uint8 transfer.
        rgb = batch['pcd_rgb'].to('cuda')
        result = operation(dict(pcd_xyz=batch['pcd_xyz'].to('cuda'), pcd_rgb=rgb))
        result[operation.target][..., 3:] = lookup[rgb.long()]
        return result

    encoder.preprocess = preprocess
    return original
