"""Original real input preprocessing; action transform supplied explicitly."""
class DataTransform:
    def __init__(self, modality, preprocessor, scaler):
        self.modality, self.preprocessor, self.scaler = modality, preprocessor, scaler

    def __call__(self, steps):
        import torch
        color = steps['pixels' if self.modality == 'pixels' else 'pcd_rgb']
        if color.dtype != torch.uint8:
            raise ValueError(f'Expected original uint8 color, got {color.dtype}')
        if self.modality == 'pcd' and not torch.isfinite(steps['pcd_xyz']).all():
            raise ValueError('Nonfinite raw point geometry')
        steps = self.preprocessor(steps)
        steps['action'] = self.scaler(steps['action'])
        return steps
