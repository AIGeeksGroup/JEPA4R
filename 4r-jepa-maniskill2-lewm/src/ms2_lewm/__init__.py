"""Isolated ManiSkill2 point-cloud LeWM model adapter."""

__all__ = ["WorldModel", "PointCloudWorldModel", "TemporalEncoder", "LeWMLoss"]


def __getattr__(name):
    if name in __all__:
        from . import model
        return getattr(model, name)
    raise AttributeError(name)
