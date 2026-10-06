"""Published Cube source arrays: retain actions and terminal rows unchanged."""
import numpy as np

def numeric_source(path):
    with np.load(path, allow_pickle=False) as source:
        actions, terminals = source['actions'], source['terminals']
    if actions.dtype != np.float32 or actions.ndim != 2 or actions.shape[1] != 5:
        raise ValueError('Official actions must remain original float32 action5')
    if terminals.shape != (len(actions),) or not np.isin(terminals, [0, 1]).all() or not terminals[-1]:
        raise ValueError('Invalid original terminal array')
    if not np.isfinite(actions).all():
        raise ValueError('Official original actions nonfinite; no NaN replacement is permitted')
    ends = np.flatnonzero(terminals)
    starts = np.r_[0, ends[:-1] + 1]
    return actions, terminals, starts, ends - starts + 1
