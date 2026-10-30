# Last3 SR completion readback

Live read at 2026-10-17T15:48:08.661Z; no NN/physics retest or raw-dataset hash.

| Checkpoint | Task1 | Task2 | Task3 | Task4 | Task5 | Overall |
| --- | --- | --- | --- | --- | --- | --- |
| 800000 | 1/50 | 1/50 | 0/50 | 0/50 | 0/50 | 2/250 = 0.8% |
| 900000 | 4/50 | 0/50 | 0/50 | 0/50 | 0/50 | 4/250 = 1.6% |
| 1000000 | 4/50 | 1/50 | 0/50 | 0/50 | 0/50 | 5/250 = 2.0% |

All3 deferred reports have actual_exit_code0, original-evaluation-body=True
and RNG_ledger_match=True. The3 CSV rows match them. There are15 ordered task
returns and3 checkpoint returns; entry ends at last3_evaluation_returned0.
The fixed binding and SHA-pinned original evaluation source establish50
statistical episodes plus1 excluded video per task:750 measured and15 video
episodes total. This is protocol/source-plus-return evidence; no independent
per-episode trajectory file is claimed.

All3 evaluation restores report153 finite leaves, Adam=checkpoint and
network.step=checkpoint+1, with model SHA matching original sidecars. The1M
template restore also passed. Current model sizes match sidecars; mtime and
ctime predate this evaluation's start. No old model was rehashed in this
readback. The original eval.csv before/after/current SHA is unchanged, all
checked source pins match, and the3 original before/after NumPy ledger hashes
match the new entry's bindings.

Trainer actual exit0; supervisor actual exit0; recorded residual_members=[];
fresh live scan of owned PGID29572 finds no non-zombie member. This
does not claim every detached service on the host is gone.

This run performs zero training updates and uses CPU evaluation of existing
state46/goal46/action5 seed0/1M models,1000-action horizon, temperature0,
GaussianNone. The new simulator has its own Generator; original actor seeds
and protocol are preserved, but stopped-run initial scenes are not reproduced.
These scores are separate from the custom paired Native protocol.

Final model/evidence backup is not claimed until archive, real network copy,
local SHA/size and extraction checks finish. Detailed receipts, exact ns stat
strings, source evaluation text and hashes are in the JSON readback.
