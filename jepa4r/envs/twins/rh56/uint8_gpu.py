# Moved from OGBench-cube@4fbb0c6:lewm_pcd/rh56_adapter/uint8_gpu.py (verbatim)
"""Experimental, default-off RGB GPU preprocessing through the existing SPT hook.

The historical full-model CPU/GPU numerical-equivalence gate failed. This flag
changes execution placement and must not be described as proven equivalent.
"""


class Uint8GPUImagePreprocessor:
    """Reuse the original deterministic image transform in FP32 after transfer."""

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.processed_batches = 0
        self.schema = None

    def __call__(self, batch):
        import torch

        pixels = batch['pixels']
        if not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.uint8:
            raise TypeError('GPU image preprocessing requires uint8 pixels')
        if pixels.device.type != 'cuda':
            raise ValueError('GPU image preprocessing requires CUDA pixels')
        if pixels.ndim not in (4, 5) or pixels.shape[-3] != 3 or any(size <= 0 for size in pixels.shape):
            raise ValueError('expected nonempty RGB NCHW or BTCHW pixels')
        input_shape = list(pixels.shape)
        flat = pixels if pixels.ndim == 4 else pixels.reshape(
            pixels.shape[0] * pixels.shape[1], *pixels.shape[-3:])
        with torch.autocast(device_type='cuda', enabled=False):
            transformed = self.preprocessor({'pixels': flat})['pixels']
        if (not isinstance(transformed, torch.Tensor) or transformed.dtype != torch.float32 or
                transformed.device.type != 'cuda' or transformed.ndim != 4 or
                transformed.shape[0] != flat.shape[0] or transformed.shape[1] != 3):
            raise ValueError('original image preprocessor must return FP32 CUDA NCHW pixels')
        batch['pixels'] = transformed if pixels.ndim == 4 else transformed.reshape(
            *pixels.shape[:2], *transformed.shape[-3:])
        self.processed_batches += 1
        if self.schema is None:
            self.schema = dict(input_shape=input_shape, input_dtype=str(pixels.dtype),
                               output_shape=list(batch['pixels'].shape), output_dtype=str(transformed.dtype),
                               device='cuda', autocast=False, route='rgb')
        return batch


def offload_image_preprocessor(dataset):
    """Move only the original first image preprocessor from CPU to SPT's GPU hook.

    Call after the baseline assigns its Compose and before constructing loaders.
    The caller explicitly enables this for the RGB CUDA route. Importing this
    module makes no change and requires neither torch nor SPT.
    """
    from stable_pretraining.data.transforms import Compose, ToImage, Resize

    cpu = dataset.transform
    if type(cpu) is not Compose or not cpu.args:
        raise ValueError('expected the original SPT CPU Compose')
    image = cpu.args[0]
    if (type(image) is not Compose or len(image.args) != 2 or
            type(image.args[0]) is not ToImage or type(image.args[1]) is not Resize or
            any(step.source != 'pixels' or step.target != 'pixels' for step in image.args)):
        raise ValueError('expected original Compose(ToImage, Resize) on pixels')
    if getattr(dataset, 'gpu_transform', None) is not None:
        raise ValueError('dataset already has a GPU transform')
    hook = Uint8GPUImagePreprocessor(image)
    dataset.transform = Compose(*cpu.args[1:])
    dataset.gpu_transform = hook
    return hook
