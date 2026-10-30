"""Portable metadata hooks; no forward computation or fixed parent counters."""
import json
import os
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from triple_training_contract import profile_binding


class TripleTrainingBinding(Callback):
    def __init__(self, cfg):
        self.binding = profile_binding(cfg)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['triple_training_binding'] = self.binding


class ResumeReadback(Callback):
    def __init__(self, output, parent):
        self.output = Path(output)
        checkpoint = torch.load(parent, map_location='cpu', weights_only=False)
        self.record = dict(parent_checkpoint=str(parent), parent_epoch=checkpoint.get('epoch'),
                           parent_global_step=checkpoint.get('global_step'),
                           random_stream_policy='seed and new loader generator; not exact RNG continuation')

    def _save(self):
        temporary = self.output.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            stream.write(json.dumps(self.record, indent=2) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.output)

    def _observe(self, trainer):
        steps = [int(value['step']) for optimizer in trainer.optimizers
                 for value in optimizer.state.values() if 'step' in value]
        return dict(epoch_index=int(trainer.current_epoch), global_step=int(trainer.global_step),
                    adam_step_min=min(steps) if steps else None, adam_step_max=max(steps) if steps else None,
                    checkpoint_callback_dirs=[str(callback.dirpath) for callback in trainer.checkpoint_callbacks])

    def on_train_start(self, trainer, pl_module):
        observed = self._observe(trainer)
        self.record['before_first_update'] = observed
        self._save()
        if observed['global_step'] != self.record['parent_global_step']:
            raise ValueError('Public global_step did not restore the full Triple parent')
        root = self.output.parent.parent.resolve()
        if any(not Path(path).resolve().is_relative_to(root) for path in observed['checkpoint_callback_dirs']):
            raise ValueError('Resumed checkpoint output escaped the new output root')

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if 'first_update' not in self.record:
            self.record['first_update'] = self._observe(trainer)
            self._save()
