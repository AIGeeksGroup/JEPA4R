"""Same frozen encoder; scoped compatibility for Python3.11+ imports only."""
from encoder import FrozenPLDM as OriginalFrozenPLDM
from py312_config_compat import compatible_defaults


class FrozenPLDM(OriginalFrozenPLDM):
    def __init__(self, asset_root):
        with compatible_defaults(asset_root):
            super().__init__(asset_root)
