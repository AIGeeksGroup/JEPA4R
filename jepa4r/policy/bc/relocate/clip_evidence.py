"""Passive clipping evidence; the original clip call alone mutates gradients."""
from contextlib import contextmanager
import math

@contextmanager
def observe_gradient_clipping(records,*,model_provider=None):
    import torch
    original=torch.nn.utils.clip_grad_norm_
    def norm(parameters,order):
        values=[torch.linalg.vector_norm(p.grad.detach(),ord=order) for p in parameters if p.grad is not None]
        if not values:return 0.
        return float(torch.linalg.vector_norm(torch.stack(values),ord=order).detach().float().cpu())
    def observed(parameters,*args,**kwargs):
        values=[parameters] if torch.is_tensor(parameters) else list(parameters)
        order=kwargs.get('norm_type',args[1] if len(args)>1 else 2.)
        if order=='inf':order=float('inf')
        model=None if model_provider is None else model_provider()
        names={} if model is None else {id(p):name for name,p in model.named_parameters()}
        encoder=[p for p in values if names.get(id(p),'').startswith('encoder.')]
        encoder_before=None if not encoder else norm(encoder,order)
        returned=original(values,*args,**kwargs)
        after=norm(values,order);encoder_after=None if not encoder else norm(encoder,order)
        before=float(returned.detach().float().cpu())
        records.append(
            dict(
                event=len(records),
                parameter_tensors=len(values),
                gradient_tensors=sum(p.grad is not None for p in values),
                max_norm=float(kwargs.get('max_norm', args[0] if args else 0.0)),
                norm_type=float(order),
                norm_before_original_return=before,
                norm_after=after,
                encoder_norm_before=encoder_before,
                encoder_norm_after=encoder_after,
                finite=math.isfinite(before) and math.isfinite(after),
                scope='Original clip_grad_norm_ return is pre-clip; added norm reads do not modify tensors or RNG',
            )
        )
        return returned
    torch.nn.utils.clip_grad_norm_=observed
    try:yield
    finally:torch.nn.utils.clip_grad_norm_=original
