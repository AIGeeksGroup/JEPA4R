"""eval.py's random-policy run with a zero action instead (the fingertip targets stay where they are): the floor of
the TriFinger planning protocol. A goal 5 rows ahead is often already met where the episode pauses (closing in,
touching, letting go, the two idle fingers of a reach), and eval.py's random policy draws from the whole action space
(+-8 cm per axis and step), which leaves those states at once: its ~0 % is not the floor. Same starts as the other runs
of a seed (eval.py draws them from the seed and the test set). eval.py itself is run unchanged.
    STABLEWM_HOME=... PYTHONPATH=.:third_party/stable-worldmodel $PY scripts/eval/eval_hold.py \
        --config-name tf_cube_pix policy=random seed=42 output.filename=...
"""
import os
import runpy
import sys

import numpy as np
import stable_worldmodel as swm


class HoldPolicy(swm.policy.RandomPolicy):
    def get_action(self, obs, **kwargs):
        return np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)


swm.policy.RandomPolicy = HoldPolicy  # eval.py builds swm.policy.RandomPolicy() for policy=random

if __name__ == '__main__':
    script = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'eval.py'))
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name='__main__')
