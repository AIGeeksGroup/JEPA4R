"""Move XYZ FP32 and RGB uint8 separately; reuse the original preprocessor math."""


def install(encoder):
    import torch
    encoder.source_paths()
    from utils import get_pcd_preprocessor
    operation = get_pcd_preprocessor(center=(.425, 0., .2), scale=4., use_rgb=True)
    operation.center = operation.center.to('cuda')
    original = encoder.preprocess

    def preprocess(batch, modality):
        if modality != '3d':
            return original(batch, modality)
        assert batch['pcd_xyz'].dtype == torch.float32 and batch['pcd_rgb'].dtype == torch.uint8
        assert batch['pcd_xyz'].shape == batch['pcd_rgb'].shape
        # Conversion, centering, scaling and concatenation remain original __call__.
        return operation(dict(pcd_xyz=batch['pcd_xyz'].to('cuda'), pcd_rgb=batch['pcd_rgb'].to('cuda')))

    encoder.preprocess = preprocess
    return original
